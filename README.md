# Facial Features Analysis Dataset

## Dataset Overview
This dataset contains facial feature coordinates and derived measurements from facial images. The dataset includes both raw coordinates of facial landmarks and calculated geometric features.

## Data Description

### Raw Features
- **image**: Image filename
- **lefteye_x, lefteye_y**: Left eye coordinates
- **righteye_x, righteye_y**: Right eye coordinates
- **nose_x, nose_y**: Nose coordinates
- **leftmouth_x, leftmouth_y**: Left corner of mouth coordinates
- **rightmouth_x, rightmouth_y**: Right corner of mouth coordinates

### Derived Measurements

#### Distances (in pixels)
- **distance_eyes**: Distance between left and right eyes
- **distance_eye_nose_left**: Distance between left eye and nose
- **distance_eye_nose_right**: Distance between right eye and nose
- **distance_nose_mouth_left**: Distance between nose and left corner of mouth
- **distance_nose_mouth_right**: Distance between nose and right corner of mouth
- **distance_mouth**: Distance between left and right corners of mouth

#### Angles (in degrees)
- **angle_nose_eyes**: Angle formed by left eye, nose, and right eye
- **angle_nose_mouth**: Angle formed by left mouth corner, nose, and right mouth corner

## Dataset Statistics

### Sample Size
- Total samples: 30 images

### Value Ranges
- Coordinates range approximately:
  - x: 38-769 pixels
  - y: 38-528 pixels
- Typical distances: 15-152 pixels
- Typical angles: 18-138 degrees


## Dependencies
- Python 3.x
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

This will:
1. Load the facial features data
2. Calculate geometric measurements
3. Perform PCA analysis
4. Generate visualizations

the data set from [text](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
This project is inspired by [Deep Learning Face Attributes in the Wild](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html)  
by Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang (ICCV 2015) "list_landmarks_celeba.txt".