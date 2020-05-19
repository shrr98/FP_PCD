import urllib
import os
from matplotlib import pyplot as plt
from transform import *
from plates import Plates

# from conf import *
# url = 'https://4g92mivec.files.wordpress.com/2014/08/plat-nomor.jpg'
url = 'https://upload.wikimedia.org/wikipedia/commons/a/a5/Plat_Nomor_Nganjuk_%283_Huruf%29.jpg'
# url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcST8y8Nr_bXfsU9jtcKh9EbVuZL1gLtBB1agjN0aVjnQ9RtBi7L&usqp=CAU'
# url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQObLn-Fniq1pj-iQcRicyeHV88FNPap62AA8Q2l6hh_t7rmyBQ&usqp=CAU'
# url = 'https://github.com/keynekassapa13/num-plate-track/blob/master/numberplates/car18.jpg?raw=true'


def resize_img(ori_img, new_height):
  height, width, _ = ori_img.shape
  new_width = int(width * float(new_height) / float(height))

  new_img = cv2.resize(ori_img, (new_width, new_height))
  return new_img


pts = []
image = None


def click_and_crop(event, x, y, flags, param):
    global pts, image
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        p = (x,y)
        pts.append(p)
        image = cv2.circle(image, p, 3, (255,0,0), 2) 

def select_plat_area(img):
    global image
    image = img.copy()
    cv2.imshow('Pilih area plat nomor', image)
    cv2.setMouseCallback("Pilih area plat nomor", click_and_crop)

    while True:
        cv2.imshow("Pilih area plat nomor", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c") and len(pts)==4:
            break
        elif key == ord('r'): # reset
            image = img.copy()
            pts.clear()
    cv2.destroyAllWindows()

    plat_img = four_point_transform(img, np.array(pts))
    return plat_img

temp_num = [f for f in os.listdir('./temp-num') if os.path.isfile(os.path.join('./temp-num', f))]
plt.rcParams["figure.figsize"] = (15,10)
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
ori_img = cv2.imdecode(image, cv2.IMREAD_COLOR)
ori_img = resize_img(ori_img=ori_img, new_height=480)
plat_img = select_plat_area(ori_img)
plat_img = resize_img(plat_img, 100)
old = Plates(plat_img, 'Car1', 0.5, temp_num)
