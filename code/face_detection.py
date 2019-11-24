import cv2 as cv
import progressbar

from utils import *


patch_size = 5

f = 721.537700
px = 609.559300
py = 172.854000
baseline = 0.5327119288
x1 = 685
x2 = 804
y1 = 181
y2 = 258


def load_color_image(filepath):
    image = cv.imread(filepath)
    return image

def load_grey_scale_image(filepath):
    image = cv.imread(filepath, 0)
    return image

left_img = load_grey_scale_image('./000020_left.jpg')
right_img = load_grey_scale_image('./000020_right.jpg')
right_color = load_color_image('./000020_right.jpg')

def show_image(img):
    cv.imshow("Showing image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def save_image(fname, img):
  cv.imwrite(fname, img)

def ssd(patch1, patch2):
    diff = patch1 - patch2
    ssd = np.sum(diff**2)
    return ssd

def nc(patch1, patch2):
    a = np.sum(patch1 * patch2)
    b = np.sum(patch1 ** 2) * np.sum(patch2 ** 2)
    c = a * 1./b
    return c
  
def scan(x1 = x1, x2 = x2, y1 = y1, y2 = y2, left_img = left_img, right_img = right_img):
  width_right = right_img.shape[1]
  depth = np.zeros((y2 - y1 + 1, x2 - x1 + 1))
  ite_num = (x2 - x1) * (y2 - y1) * (width_right - 2 * patch_size)
  bar = progressbar.ProgressBar(maxval=ite_num+1,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  count = 0
  myobj = plt.imshow(right_color)
  for x in range(x1, x2 + 1):
    for y in range(y1, y2 + 1):
      target_patch = left_img[y - patch_size: y + patch_size, x - patch_size: x + patch_size]
      
      scores = []
      for x2 in range(patch_size, width_right - patch_size):
        source_patch = right_img[y - patch_size: y + patch_size, x2 - patch_size: x2 + patch_size]
        score = nc(source_patch, target_patch)
        scores.append(score)
        count += 1
        bar.update(count)
      fx = np.argmin(scores)
      right_color[y, fx] = np.array([0,255,0])
      myobj.set_data(right_color)
      plt.draw()
  
      

  
if __name__ == "__main__":
  scan()
    
