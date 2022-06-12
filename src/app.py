import base64
import eel
import os
import cv2
import numpy as np
import time

current_file_extension = ".png"
start_time = time.time()
# base64轉np image

def data_uri_to_cv2_img(uri):
    global start_time
    start_time = time.time()
    uri_spt = uri.split(',')
    if(len(uri_spt) > 1):
        encoded_data = uri_spt[1]
    else:
        encoded_data = uri_spt[0]

    img = base64.b64decode(encoded_data)
    npimg = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    #print("base 64 decode Time --- %s seconds ---" % (time.time() - start_time))
    return image

# 圖片轉base64


def img_to_base64(img):
    #b64_str = cv2.imencode(".png", img)[1].tobytes()
    b64_str = cv2.imencode(current_file_extension, img)[1].tobytes()
    blob = base64.b64encode(b64_str)
    blob = blob.decode('utf-8')
    print("process Time --- %s seconds ---" % (time.time() - start_time))
    return blob


@eel.expose
def setFileExtension(ext):
    global current_file_extension
    current_file_extension = "."+ext


@eel.expose
def getBlueChannels(url):
    img = data_uri_to_cv2_img(url)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    b_img = img_to_base64(b)
    g_img = img_to_base64(g)
    r_img = img_to_base64(r)
    eel.SetImgWithID(r_img, "red_channel_img")()
    eel.SetImgWithID(g_img, "green_channel_img")()
    eel.SetImgWithID(b_img, "blue_channel_img")()


@eel.expose
def histGraph(url):
    img = data_uri_to_cv2_img(url)

    # 單色
    if img.ndim != 3:
        #hist = cv2.calcHist( [img], [0], None, [256], [0,256] )
        b = cv2.calcHist([img], [0], None, [256], [0, 256])
        b = b.astype("float").flatten().tolist()

    # 彩色
    else:
        b = cv2.calcHist([img], [0], None, [256], [0, 256])
        g = cv2.calcHist([img], [1], None, [256], [0, 256])
        r = cv2.calcHist([img], [2], None, [256], [0, 256])
        b = b.astype("float").flatten().tolist()
        g = g.astype("float").flatten().tolist()
        r = r.astype("float").flatten().tolist()

    eel.UpdateGarphData("histogram", b, g, r)()

# 輸入圖片更新直方圖


def histGraphByImg(img):

    # 單色
    if img.ndim != 3:
        b = cv2.calcHist([img], [0], None, [256], [0, 256])
        b = b.astype("float").flatten().tolist()
    # 彩色
    else:
        b = cv2.calcHist([img], [0], None, [256], [0, 256])
        g = cv2.calcHist([img], [1], None, [256], [0, 256])
        r = cv2.calcHist([img], [2], None, [256], [0, 256])
        b = b.astype("float").flatten().tolist()
        g = g.astype("float").flatten().tolist()
        r = r.astype("float").flatten().tolist()
    eel.UpdateGarphData("histogram", b, g, r)()


@eel.expose
def equalizeHist(url):
    img = data_uri_to_cv2_img(url)
    g = img.copy()
    # 單色
    if g.ndim != 3:
        g = cv2.equalizeHist(img)
    # 彩色
    else:
        for i in range(0, 3):
            g[:, :, i] = cv2.equalizeHist(img[:, :, i])

    # histGraphByImg(g)
    blob = img_to_base64(g)
    eel.SetImg(blob)()


# 轉灰階
@eel.expose
def toGray(url):
    img = data_uri_to_cv2_img(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

# 轉灰階
@eel.expose
def negativeFlim(url):
    img = data_uri_to_cv2_img(url)
    img2= 255-img
    blob = img_to_base64(img2)
    eel.SetImg(blob)()
    
#顏色裁切
@eel.expose
def HSV_color_segmentation( url, H1, H2, S1, S2, V1, V2 ):
    img = data_uri_to_cv2_img(url)
    H1=int(H1)
    H2=int(H2)
    S1=int(S1)
    S2=int(S2)
    V1=int(V1)
    V2=int(V2)
    hsv = np.int16(cv2.cvtColor( img, cv2.COLOR_BGR2HSV ))
	#邏輯運算找該角度內的值
    h=np.logical_and(hsv[:,:,0]>=(H1/2),hsv[:,:,0]<=(H2/2))
    s=np.logical_and(hsv[:,:,1]>=(S1/100*255),hsv[:,:,1]<=(S2/100*255))
    v=np.logical_and(hsv[:,:,2]>=(V1/100*255),hsv[:,:,2]<=(V2/100*255))
    mask=np.logical_and(h,s,v)*1
    img[:,:,0]=img[:,:,0]*mask
    img[:,:,1]=img[:,:,1]*mask
    img[:,:,2]=img[:,:,2]*mask
    
    blob = img_to_base64(img)
    eel.SetImg(blob)()
    
# 轉HSL


@eel.expose
def toHSV(url):
    img = data_uri_to_cv2_img(url)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

# 轉CMYK


@eel.expose
def toCMYK(url):
    img = data_uri_to_cv2_img(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blob = img_to_base64(img)
    eel.SetImg(blob)()


@eel.expose
def setHSV_V(url, value):
    img = data_uri_to_cv2_img(url)
    value = int(value)
    hsv = np.int16(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    hsv[:, :, 2] = hsv[:, :, 2] * value / 100

    hsv = np.uint8(hsv)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blob = img_to_base64(bgr)
    eel.SetImg(blob)()


@eel.expose
def setHSV_S(url, value):
    img = data_uri_to_cv2_img(url)
    value = int(value)
    hsv = np.int16(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    hsv[:, :, 1] = hsv[:, :, 1] * value / 100

    hsv = np.uint8(hsv)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blob = img_to_base64(bgr)
    eel.SetImg(blob)()


@eel.expose
def setHSV_H(url, value):
    img = data_uri_to_cv2_img(url)
    value = int(value)
    hsv = np.int16(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    hsv[:, :, 0] = (hsv[:, :, 0]+(value/2)) % 180

    hsv = np.uint8(hsv)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blob = img_to_base64(bgr)
    eel.SetImg(blob)()

#曝光
@eel.expose
def gamma(url , v):
    img = data_uri_to_cv2_img(url)
    value = float(v)
    g=(img/255)**value  #(數值轉0~1)**gamma
    g=np.uint8(g*255) #轉回0~255
    blob = img_to_base64(g)
    eel.SetImg(blob)()

#對比
@eel.expose
def beta(url , a,b):
    img = data_uri_to_cv2_img(url)
    a = float(a)
    b = float(b)
    import scipy.special as special
    g = np.uint8(( special.betainc( a, b, img/255 ))*255)
    blob = img_to_base64(g)
    eel.SetImg(blob)()
 
    
# 高斯模糊
@eel.expose
def gaussianblur(url, ksize, sigma):
    img = data_uri_to_cv2_img(url)
    ksize = int(ksize)
    sigma = int(sigma)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

# 中值濾波


@eel.expose
def mddianFilter(url, ksize):
    img = data_uri_to_cv2_img(url)
    ksize = int(ksize)
    img = cv2.medianBlur(img, ksize)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

# 雙邊濾波


@eel.expose
def bilateralFilter(url, _d, _sc, _ss):
    img = data_uri_to_cv2_img(url)
    d = int(_d)
    sc = float(_sc)
    ss = float(_ss)
    img = cv2.bilateralFilter(img, d, sc, ss)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

# 銳化


@eel.expose
def unsharpMasking(url, k_size):
    img = data_uri_to_cv2_img(url)
    k_size = float(k_size)
    f_avg = cv2.GaussianBlur(img, (15, 15), 0)
    g_mask = (img.astype(int))-(f_avg.astype(int))
    g = np.uint8(np.clip(img+k_size*g_mask, 0, 255))

    blob = img_to_base64(g)
    eel.SetImg(blob)()


# Resize
@eel.expose
def resize(url, ratio, opt):
    img = data_uri_to_cv2_img(url)
    ratio = float(ratio)

    x, y = img.shape[:2]
    x = int(x*ratio)
    y = int(y*ratio)
    if opt == "linear":
        img = cv2.resize(img, (y, x), interpolation=cv2.INTER_LINEAR)
    elif opt == "nearst":
        img = cv2.resize(img, (y, x), interpolation=cv2.INTER_NEAREST)
    elif opt == "cubic":
        img = cv2.resize(img, (y, x), interpolation=cv2.INTER_CUBIC)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

@eel.expose
def horizontalFlip(url):
    img = data_uri_to_cv2_img(url)
    image2 = cv2.flip(img, 1)    
    blob = img_to_base64(image2)
    eel.SetImg(blob)()
    
@eel.expose
def verticalFlip(url):
    img = data_uri_to_cv2_img(url)
    image2 = cv2.flip(img, 0)  
    blob = img_to_base64(image2)
    eel.SetImg(blob)()  

@eel.expose
def rotate(url, angle):
    angle = int(angle)
    img = data_uri_to_cv2_img(url)
    x, y = img.shape[1], img.shape[0]
    rotate_mat = cv2.getRotationMatrix2D(
        (x/2, y/2),  # 中心位置
        angle,
        scale=1
    )
    # 計算旋轉後的bound大小
    mcos = abs(rotate_mat[0][0])
    msin = abs(rotate_mat[0][1])
    bound_x = int(mcos*x+msin*y)
    bound_y = int(msin*x+mcos*y)
    # 編輯偏移 (減去舊圖的center使圖片回到中心，再加上新的offset)
    rotate_mat[0][2] += bound_x/2 - x/2
    rotate_mat[1][2] += bound_y/2 - y/2
    img_r = cv2.warpAffine(  # 處理影象的偏移
        img,
        rotate_mat,
        (bound_x, bound_y))  # 輸出size
    blob = img_to_base64(img_r)
    eel.SetImg(blob)()

# 邊緣提取


@eel.expose
def grad(url, k_type):
    img = data_uri_to_cv2_img(url)

    if k_type == "robers":
        kx = np.array([[-1, 0], [0, 1]])
        ky = np.array([[0, -1], [1, 0]])

    elif k_type == "prewitt":
        kx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    elif k_type == "sobel":
        kx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    elif k_type == "robinson":
        kx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    elif k_type == "kirsh":
        kx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    grad_x = cv2.filter2D(img, cv2.CV_32F, kx)
    grad_y = cv2.filter2D(img, cv2.CV_32F, ky)

    magnitude = abs(grad_x) + abs(grad_y)
    g = np.uint8(np.clip(magnitude, 0, 255))

    blob = img_to_base64(g)
    eel.SetImg(blob)()


# 邊緣保留
@eel.expose
def edgePreserve(url):
    img = data_uri_to_cv2_img(url)
    img2 = cv2.edgePreservingFilter(img, flags=1, sigma_s=150, sigma_r=0.8)

    blob = img_to_base64(img2)
    eel.SetImg(blob)()

# 鉛筆風


@eel.expose
def pencilStyle(url):
    img = data_uri_to_cv2_img(url)
    gray, img2 = cv2.pencilSketch(img)

    blob = img_to_base64(img2)
    eel.SetImg(blob)()

# 風格化


@eel.expose
def stylization(url):
    img = data_uri_to_cv2_img(url)
    img2 = cv2.stylization(img)

    blob = img_to_base64(img2)
    eel.SetImg(blob)()

# 毛邊模糊


@eel.expose
def fuzzy(url):
    f = data_uri_to_cv2_img(url)
    W = 20
    nr, nc = f.shape[:2]
    y, x = np.mgrid[0:nr, 0:nc]
    x2 = x+np.random.randint(-W//2, W//2, size=(nr, nc))
    y2 = y+np.random.randint(-W//2, W//2, size=(nr, nc))
    x2 = np.clip(x2, 0, nc-1)
    y2 = np.clip(y2, 0, nr-1)
    g = cv2.remap(f, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR)

    blob = img_to_base64(g)
    eel.SetImg(blob)()

# 波紋模糊


@eel.expose
def ripple(url, amp, pid):
    img = data_uri_to_cv2_img(url)
    amp = float(amp)
    pid = int(pid)

    nr, nc = img.shape[:2]
    y, x = np.mgrid[0:nr, 0:nc]  # 沒有考慮原點位移
    x1 = x+amp * np.sin(x / pid)
    y1 = y+amp * np.sin(y / pid)

    x2 = np.clip(x1, 0, nc-1)
    y2 = np.clip(y1, 0, nr-1)
    img2 = cv2.remap(img, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR)

    blob = img_to_base64(img2)
    eel.SetImg(blob)()

# 放射像素風格


@eel.expose
def radialPixelation(url, delta_r, delta_theta):
    img = data_uri_to_cv2_img(url)
    delta_r = int(delta_r)
    delta_theta = float(delta_theta)

    nr, nc = img.shape[:2]
    # 找到中心點座標
    y0, x0 = (nr+1) // 2, (nc+1) // 2
    # mgrid [start : end : step] 產生水平、垂直的start~end的表格
    y, x = np.mgrid[-y0:nr-y0, -x0:nc-x0]
    # 算r
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y, x)
    # 每隔delta_r 距離的像素方塊
    r2 = r - r % delta_r
    # 每隔delta_theta 角度的像素方塊
    theta2 = theta - theta % (np.radians(delta_theta))
    # 極座標轉回笛卡兒座標，且位移原點回左上角
    x2 = np.clip(x0+r2*np.cos(theta2), 0, nc-1)
    y2 = np.clip(y0+r2*np.sin(theta2), 0, nr-1)
    # remap（）圖像的重映射，可以把一幅圖像中某位置的圖元放置到另一個圖片指定位置的過程。
    img2 = cv2.remap(img, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR)

    blob = img_to_base64(img2)
    eel.SetImg(blob)()


# 放射漣漪模糊
@eel.expose
def radialRipple(url, amp, pid):
    img = data_uri_to_cv2_img(url)
    amp = float(amp)
    pid = int(pid)
    nr, nc = img.shape[:2]
    # uv offset
    y0, x0 = (nr+1) // 2, (nc+1) // 2
    y, x = np.mgrid[-y0:nr-y0, -x0:nc-x0]  # 原點位移

    # 轉極座標
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y, x)

    # 波面位移
    r2 = r + np.sin(r/pid)*amp

    x2 = np.clip(x0+r2*np.cos(theta), 0, nc-1)
    y2 = np.clip(y0+r2*np.sin(theta), 0, nr-1)

    img2 = cv2.remap(img, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR)
    blob = img_to_base64(img2)
    eel.SetImg(blob)()


@eel.expose
def fisheye(url):
    img = data_uri_to_cv2_img(url)
    nr, nc = img.shape[:2]
    # uv offset
    y0, x0 = (nr+1) // 2, (nc+1) // 2
    y, x = np.mgrid[-y0:nr-y0, -x0:nc-x0]  # 原點位移

    r = np.sqrt(x**2+y**2)
    #R= np.sqrt(nr**2+nc**2)/2
    R = np.max(r)
    theta = np.arctan2(y, x)

    rp = r*r/R

    xp = np.clip(x0 + rp*np.cos(theta), 0, nr-1)
    yp = np.clip(y0 + rp*np.sin(theta), 0, nc-1)

    img2 = cv2.remap(img, np.float32(xp), np.float32(yp), cv2.INTER_LINEAR)
    blob = img_to_base64(img2)
    eel.SetImg(blob)()


@eel.expose
def twirl(url, k):
    k = float(k)
    img = data_uri_to_cv2_img(url)
    nr, nc = img.shape[:2]
    # uv offset
    y0, x0 = (nr+1) // 2, (nc+1) // 2
    y, x = np.mgrid[-y0:nr-y0, -x0:nc-x0]  # 原點位移

    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y, x)

    t2 = theta + r/k
    xp = np.clip(x0 + r*np.cos(t2), 0, nr-1)
    yp = np.clip(y0 + r*np.sin(t2), 0, nc-1)

    img2 = cv2.remap(img, np.float32(xp), np.float32(yp), cv2.INTER_LINEAR)
    blob = img_to_base64(img2)
    eel.SetImg(blob)()


@eel.expose
def motionBlur(url, l, a):

    leng = int(l)
    ang = float(a)
    img = data_uri_to_cv2_img(url)
    #nr,nc= img.shape[:2]
    filter = np.zeros([leng, leng])
    x0, y0 = leng//2, leng//2
    x_len = round(x0*np.cos(np.radians(ang)))
    y_len = round(y0*np.sin(np.radians(ang)))

    x1, y1 = int(x0-x_len), int(y0-y_len)
    x2, y2 = int(x0+x_len), int(y0+y_len)
    cv2.line(filter, (y1, x1), (y2, x2), (1, 1, 1))
    filter /= np.sum(filter)
    g = cv2.filter2D(img, -1, filter)

    blob = img_to_base64(g)
    eel.SetImg(blob)()


@eel.expose
def radicalBlur(url, k):
    k = int(k)
    img = data_uri_to_cv2_img(url)
    height, width = img.shape[:2]

    # uv offset
    y0, x0 = (height+1) // 2, (width+1) // 2
    y, x = np.mgrid[-y0:height-y0, -x0:width-x0]  # 原點位移

    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y, x)
    # 灰階圖處理
    if img.ndim < 3:
        sum = np.zeros((height, width))
    else:
        sum = np.zeros((height, width, img.ndim))

    for _k in range(-k, k+1):
        phi = theta + np.radians(_k)
        xp = np.clip(x0 + r*np.cos(phi), 0, width-1).astype(int)
        yp = np.clip(y0 + r*np.sin(phi), 0, height-1).astype(int)
        if img.ndim < 3:
            sum += img[yp, xp]
        else:
            sum += img[yp, xp, :]
        # sum[yp,xp,:]+=img[yp,xp,:] #ERROR

    img2 = (sum/(k*2+1)).astype("uint8")
    blob = img_to_base64(img2)
    eel.SetImg(blob)()


@eel.expose
def faceDetection(url, scale, minNei, method):
    scale = float(scale)
    minNei = int(minNei)
    print(scale,minNei)

    img = data_uri_to_cv2_img(url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if(method == "haar"):
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
    elif(method == "lbp"):
        # lbp比較快，但比較不準
        face_cascade = cv2.CascadeClassifier('lbpcascade_frontalcatface.xml')
    else:
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
    # detectMultiScale (圖片、縮放scale、重複被框出的次數閥值)
    faces = face_cascade.detectMultiScale(gray, scale, minNei)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    blob = img_to_base64(img)
    eel.SetImg(blob)()


@eel.expose
def useChannel(url, isR, isG, isB):
    isR = bool(isR)
    isG = bool(isG)
    isB = bool(isB)
    img = data_uri_to_cv2_img(url)

    g = np.zeros(img.shape)

    if(isR and isG and isB):
        eel.SetImg(url, False, False)()
        return

    if isR:
        g[:, :, 2] = img[:, :, 2]
    if isB:
        g[:, :, 0] = img[:, :, 0]
    if isG:
        g[:, :, 1] = img[:, :, 1]

    blob = img_to_base64(g)
    eel.SetImg(blob, False, False)()

@eel.expose
def overlayImage(url,layerUrl,opt):
    
    
    img = data_uri_to_cv2_img(url)
    img_overlay = data_uri_to_cv2_img(layerUrl)    
    
    h, w = img_overlay.shape[:2]    
    hh, ww = img.shape[:2]    
    xMin= min(ww,w)
    yMin= min(hh,h)
    yoff = np.clip( round((hh-h)/2  ),0,hh)
    xoff =np.clip( round((ww-w)/2 ),0,ww)    

    yMax= np.clip(yoff+h, 0 , hh)
    xMax= np.clip(xoff+w , 0 , ww)

    clipped_overlayImg=img_overlay[:yMin,:xMin] #裁減的大小
    clipped_orginImg=img[yoff:yMax, xoff:xMax] #被覆蓋的原圖大小
    if opt=="normal":
        result = clipped_overlayImg
    elif opt=="darken":
        result= np.minimum(clipped_orginImg,clipped_overlayImg)
    elif opt=="brighter":
        result= np.maximum(clipped_orginImg,clipped_overlayImg)
    elif opt=="screen":
        up=np.float32(clipped_overlayImg)/255.0
        down=np.float32(clipped_orginImg)/255.0
        result = up+down-(up*down)
        result = np.uint8(np.clip(result *255 ,0,255))
    elif opt=="overlay":
        up=np.float32(clipped_overlayImg)/255.0
        down=np.float32(clipped_orginImg)/255.0
        result = down+(up-0.5)*(1-(np.abs(down-0.5)/0.5))
        result = np.uint8(np.clip(result *255 ,0,255))
        
    f=img.copy()
    f[yoff:yMax, xoff:xMax] = result
    #result[yoff:yMax, xoff:xMax] = img_overlay[:yMin,:xMin]
    
    blob = img_to_base64(f)
    eel.SetImg(blob)()
    
    
    
    
# 開啟網頁
eel.init(f'{os.path.dirname(os.path.realpath(__file__))}/web')
#eel.start('main.html', mode='chrome-app',port=8000,cmdline_args=['--start-fullscreen', '--browser-startup-dialog'])  # 網頁 (app模式)
eel.start('main.html', mode='edge',port=8080,cmdline_args=['--start-fullscreen'])  # 網頁 (app模式)
