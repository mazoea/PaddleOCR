# Understanding Shrink Maps in PaddleOCR DB (Differentiable Binarization)

## Overview
This document explains step-by-step how the `forward()` function works in PaddleOCR's DB (Differentiable Binarization) detection model, with a focus on **shrink maps** - their creation, purpose, and usage.

---

## Table of Contents
1. [The Big Picture: What is DB?](#the-big-picture)
2. [Forward Pass Flow](#forward-pass-flow)
3. [Shrink Maps: Deep Dive](#shrink-maps-deep-dive)
4. [Why Shrink Maps?](#why-shrink-maps)
5. [Complete Training Pipeline](#complete-training-pipeline)

---

## The Big Picture: What is DB?

**DB (Differentiable Binarization)** is a text detection algorithm that predicts three types of maps:
1. **Shrink Map** (Probability Map) - Shows where text regions are
2. **Threshold Map** - Adaptive thresholds for binarization
3. **Binary Map** - Final text/non-text segmentation (computed during training)

The key innovation: Instead of using fixed thresholds, DB learns adaptive thresholds for each pixel.

---

## Forward Pass Flow

### Step 1: Model Architecture (`BaseModel.forward()`)

Location: `ppocr/modeling/architectures/base_model.py`

```python
def forward(self, x, data=None):
    # x = input image [batch, 3, height, width]
    
    # 1. Transform (optional - not used in DB)
    if self.use_transform:
        x = self.transform(x)
    
    # 2. Backbone - Extracts features
    #    Example: PPLCNetV3 extracts multi-scale features
    if self.use_backbone:
        x = self.backbone(x)
        # Output: Feature maps [batch, channels, H/4, W/4]
    
    # 3. Neck - Fuses multi-scale features
    #    Example: RSEFPN combines features from different scales
    if self.use_neck:
        x = self.neck(x)
        # Output: Enhanced features [batch, 256, H/4, W/4]
    
    # 4. Head - Produces prediction maps
    if self.use_head:
        x = self.head(x, targets=data)
        # Output: Dictionary with 'maps' key
    
    return x
```

**Key Points:**
- Input image is downsampled by backbone (typically 4x)
- Features are enhanced by neck (FPN-like structure)
- Head produces the final prediction maps

---

### Step 2: DB Head (`DBHead.forward()`)

Location: `ppocr/modeling/heads/det_db_head.py`

```python
class DBHead(nn.Layer):
    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k  # Amplification factor for step function
        self.binarize = Head(in_channels, **kwargs)  # Predicts shrink map
        self.thresh = Head(in_channels, **kwargs)    # Predicts threshold map
    
    def step_function(self, x, y):
        # Differentiable approximation of step function
        # x: shrink_maps, y: threshold_maps
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))
    
    def forward(self, x, targets=None):
        # Step 1: Predict shrink map (probability of text)
        shrink_maps = self.binarize(x)
        # Shape: [batch, 1, H, W]
        # Values: [0, 1] - probability of being text
        
        # During inference, only return shrink map
        if not self.training:
            return {"maps": shrink_maps}
        
        # Step 2: Predict threshold map (adaptive threshold for each pixel)
        threshold_maps = self.thresh(x)
        # Shape: [batch, 1, H, W]
        # Values: [0, 1] - threshold for binarization
        
        # Step 3: Apply differentiable binarization
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        # Shape: [batch, 1, H, W]
        # This is: 1 / (1 + exp(-k * (shrink_maps - threshold_maps)))
        # When shrink_maps > threshold_maps: binary_maps → 1
        # When shrink_maps < threshold_maps: binary_maps → 0
        
        # Step 4: Concatenate all three maps
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        # Shape: [batch, 3, H, W]
        
        return {"maps": y}
```

**Key Understanding:**
- **Channel 0**: Shrink map (probability map)
- **Channel 1**: Threshold map (adaptive thresholds)
- **Channel 2**: Binary map (final segmentation)

---

### Step 3: Head Architecture (Upsampling)

Location: `ppocr/modeling/heads/det_db_head.py`

```python
class Head(nn.Layer):
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()
        
        # 1. Conv + BatchNorm + ReLU (reduce channels)
        self.conv1 = nn.Conv2D(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv_bn1 = nn.BatchNorm(in_channels // 4, act="relu")
        
        # 2. Transposed Conv (upsample 2x)
        self.conv2 = nn.Conv2DTranspose(in_channels // 4, in_channels // 4, 
                                        kernel_size=2, stride=2)
        self.conv_bn2 = nn.BatchNorm(in_channels // 4, act="relu")
        
        # 3. Transposed Conv (upsample 2x again) + Sigmoid
        self.conv3 = nn.Conv2DTranspose(in_channels // 4, 1, 
                                        kernel_size=2, stride=2)
    
    def forward(self, x, return_f=False):
        # Input: [batch, 256, H/4, W/4]
        x = self.conv1(x)
        x = self.conv_bn1(x)  # [batch, 64, H/4, W/4]
        
        x = self.conv2(x)
        x = self.conv_bn2(x)  # [batch, 64, H/2, W/2]
        
        x = self.conv3(x)     # [batch, 1, H, W]
        x = F.sigmoid(x)      # Values in [0, 1]
        return x
```

**Upsampling Process:**
- Input feature: 1/4 of original image size
- After 2 transposed convolutions: back to original size
- Sigmoid ensures output is in [0, 1] range

---

## Shrink Maps: Deep Dive

### What is a Shrink Map?

A **shrink map** is a binary mask where:
- **Value = 1**: Pixel is inside a text region (but not at the boundary)
- **Value = 0**: Pixel is outside text or at the boundary

The "shrinking" creates a margin around text boundaries to:
1. Handle touching text instances
2. Provide clear separation between adjacent text
3. Make boundaries easier to learn

### How are Shrink Maps Created?

Location: `ppocr/data/imaug/make_shrink_map.py`

```python
class MakeShrinkMap(object):
    def __init__(self, min_text_size=8, shrink_ratio=0.4, **kwargs):
        self.min_text_size = min_text_size
        self.shrink_ratio = 0.4  # Shrink to 40% of original
    
    def __call__(self, data):
        image = data["image"]
        text_polys = data["polys"]  # Ground truth polygons
        
        h, w = image.shape[:2]
        gt = np.zeros((h, w), dtype=np.float32)  # Shrink map
        mask = np.ones((h, w), dtype=np.float32)  # Valid region mask
        
        for i, polygon in enumerate(text_polys):
            # Step 1: Create Shapely polygon
            polygon_shape = Polygon(polygon)
            
            # Step 2: Calculate shrink distance
            # Formula: distance = Area * (1 - ratio²) / Perimeter
            distance = (polygon_shape.area * (1 - np.power(0.4, 2)) 
                       / polygon_shape.length)
            
            # Step 3: Shrink polygon using PyCLipper
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND, 
                          pyclipper.ET_CLOSEDPOLYGON)
            shrunk = padding.Execute(-distance)  # Negative distance = shrink
            
            # Step 4: Fill shrunk polygon with 1
            for each_shrink in shrunk:
                shrink = np.array(each_shrink).reshape(-1, 2)
                cv2.fillPoly(gt, [shrink.astype(np.int32)], 1)
        
        data["shrink_map"] = gt      # Ground truth shrink map
        data["shrink_mask"] = mask   # Valid training region
        return data
```

**Shrink Calculation:**

```
Original polygon: ████████████
                 ████████████
                 ████████████

After shrinking (ratio=0.4):
                 ████████████
                 ██████████  <- Inner region
                 ████████████
```

**The Math:**
- `distance = Area × (1 - 0.4²) / Perimeter`
- `distance = Area × 0.84 / Perimeter`
- Larger shapes shrink more in absolute pixels
- Shape is preserved proportionally

---

## Why Shrink Maps?

### Problem 1: Touching Text
Without shrinking, adjacent text instances might merge:
```
Original:     "HELLO" "WORLD"
Without shrink: ████████████████  <- One blob
With shrink:    ████    ████      <- Two separate regions
```

### Problem 2: Boundary Ambiguity
Exact boundaries are hard to annotate and predict:
```
Ground Truth:  ████████  <- Where exactly is the edge?
Shrink Map:     ██████   <- Clear inner region
Threshold Map:  ████████ <- Learns to expand back
```

### Problem 3: Training Stability
- Shrinking creates easier-to-learn targets
- Network predicts conservative (inner) regions
- Threshold map learns to expand to actual boundaries

---

## Complete Training Pipeline

### Data Flow During Training:

```
1. DATA PREPARATION (make_shrink_map.py)
   Input: Image + Ground truth polygons
   Output: 
   - batch[0]: Image [batch, 3, H, W]
   - batch[1]: Threshold map (target) [batch, H, W]
   - batch[2]: Threshold mask [batch, H, W]
   - batch[3]: Shrink map (target) [batch, H, W]  ← GROUND TRUTH
   - batch[4]: Shrink mask [batch, H, W]

2. FORWARD PASS (model)
   preds = model(images)
   preds["maps"]: [batch, 3, H, W]
   - Channel 0: Predicted shrink map
   - Channel 1: Predicted threshold map
   - Channel 2: Predicted binary map

3. LOSS CALCULATION (det_db_loss.py)
   ```python
   def forward(self, predicts, labels):
       predict_maps = predicts["maps"]
       (label_threshold_map, label_threshold_mask,
        label_shrink_map, label_shrink_mask) = labels[1:]
       
       # Extract predicted maps
       shrink_maps = predict_maps[:, 0, :, :]      # Predicted
       threshold_maps = predict_maps[:, 1, :, :]   # Predicted
       binary_maps = predict_maps[:, 2, :, :]      # Predicted
       
       # Loss 1: Shrink map loss (BCE or Dice)
       loss_shrink_maps = self.bce_loss(
           shrink_maps,        # Predicted
           label_shrink_map,   # Ground truth (from MakeShrinkMap)
           label_shrink_mask   # Valid region
       )
       
       # Loss 2: Threshold map loss (L1)
       loss_threshold_maps = self.l1_loss(
           threshold_maps,
           label_threshold_map,
           label_threshold_mask
       )
       
       # Loss 3: Binary map loss (Dice)
       loss_binary_maps = self.dice_loss(
           binary_maps,        # Predicted (from step function)
           label_shrink_map,   # Ground truth
           label_shrink_mask
       )
       
       # Total loss
       loss_all = (5 * loss_shrink_maps + 
                   10 * loss_threshold_maps + 
                   loss_binary_maps)
       
       return {"loss": loss_all, ...}
   ```

4. BACKPROPAGATION
   - Gradients flow through step_function (differentiable!)
   - Both binarize and thresh heads are updated
   - Model learns to predict good shrink maps AND thresholds

5. INFERENCE
   ```python
   def forward(self, x, targets=None):
       shrink_maps = self.binarize(x)
       if not self.training:
           return {"maps": shrink_maps}  # Only shrink map!
   ```
   - Only shrink map is used during inference
   - Post-processing expands regions using threshold
```

---

## Key Insights

### 1. **Shrink Map is the Core Prediction**
   - Primary output during inference
   - Represents probability of text (inner region)
   - Values: 0 (no text) to 1 (definitely text)

### 2. **Three Maps Work Together During Training**
   - **Shrink map**: Predicts conservative text regions
   - **Threshold map**: Learns adaptive expansion
   - **Binary map**: Combines both for final segmentation

### 3. **Differentiable Binarization is Key**
   ```python
   binary = 1 / (1 + exp(-k * (shrink - threshold)))
   ```
   - Smooth approximation of hard threshold
   - Allows gradient flow
   - Parameter k=50 controls sharpness

### 4. **Why Shrink?**
   - **Separation**: Handles touching text instances
   - **Clarity**: Easier-to-learn inner regions
   - **Expansion**: Threshold map learns to grow back to boundaries
   - **Robustness**: More stable training

### 5. **From Shrink to Final Detection**
   ```
   Shrink Map (0.6)  +  Threshold (0.3)  =  Binary (1)
   Shrink Map (0.2)  +  Threshold (0.3)  =  Binary (0)
   ```
   The binary map effectively asks:
   "Is the shrink probability higher than the threshold?"

---

## Visualization Example

```
Original Image:
┌─────────────────┐
│                 │
│  ████████       │  <- "HELLO" text
│  ████████       │
│                 │
└─────────────────┘

Ground Truth Shrink Map (from MakeShrinkMap):
┌─────────────────┐
│                 │
│   ██████        │  <- Shrunk by 40%
│   ██████        │
│                 │
└─────────────────┘

Predicted Maps (from model):
Channel 0 (Shrink):    Channel 1 (Threshold):  Channel 2 (Binary):
┌────────────┐         ┌────────────┐          ┌────────────┐
│            │         │            │          │            │
│  ░░████░░  │         │  ▓▓████▓▓  │          │  ████████  │
│  ░░████░░  │         │  ▓▓████▓▓  │          │  ████████  │
│            │         │            │          │            │
└────────────┘         └────────────┘          └────────────┘
Low prob at edges      Lower threshold         Final result
                       at edges (easier        (expanded back)
                       to include)
```

---

## Summary

1. **`forward()` in BaseModel**: Passes image through Backbone → Neck → Head
2. **`forward()` in DBHead**: Produces 3 maps (shrink, threshold, binary)
3. **Shrink maps are created** by MakeShrinkMap during data loading by shrinking ground truth polygons
4. **Purpose**: Separate touching text, provide clear training targets, enable adaptive thresholds
5. **Training**: Model learns to predict both shrink maps and adaptive thresholds
6. **Inference**: Only shrink map is used, post-processing handles final detection

The beauty of DB is that it learns both WHAT (shrink map) and HOW (threshold map) to detect text, making it adaptive and robust!

---

## References

- **Paper**: Real-time Scene Text Detection with Differentiable Binarization (DB)
- **Code**: `ppocr/modeling/heads/det_db_head.py`
- **Data**: `ppocr/data/imaug/make_shrink_map.py`
- **Loss**: `ppocr/losses/det_db_loss.py`
- **Training**: `tools/program.py`
