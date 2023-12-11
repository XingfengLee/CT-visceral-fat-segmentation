import numpy as np 

def cal_dice_coef(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Shape mismatch: img1 and img2 must have to be of the same size.")
        else:
            lenIntersection = 0
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    if ( np.array_equal(img1[i][j],img2[i][j]) ):
                        lenIntersection+=1
            lenimg1 = img1.shape[0]*img1.shape[1]
            lenimg2 = img2.shape[0]*img2.shape[1]  
            value   = (2. * lenIntersection  / (lenimg1 + lenimg2))
        return value