import base64
import eel
import os
import cv2
import numpy as np

# base64轉np image


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    img = base64.b64decode(encoded_data)
    npimg = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return image

# 圖片轉base64
def img_to_base64(img):
    b64_str = cv2.imencode(".png", img)[1].tobytes()
    blob = base64.b64encode(b64_str)
    blob = blob.decode('utf-8')
    return blob

@eel.expose
def histGraph(url):
    img = data_uri_to_cv2_img(url)    
    
    # 單色
    if img.ndim !=3:
        #hist = cv2.calcHist( [img], [0], None, [256], [0,256] )            
        b=cv2.calcHist( [img], [0], None, [256], [0,256] )            
        b= b.astype("float").flatten().tolist()
        
    # 彩色       
    else:
        b=cv2.calcHist( [img], [0], None, [256], [0,256] )            
        g=cv2.calcHist( [img], [1], None, [256], [0,256] )            
        r=cv2.calcHist( [img], [2], None, [256], [0,256] )            
        b= b.astype("float").flatten().tolist()
        g= g.astype("float").flatten().tolist()
        r= r.astype("float").flatten().tolist()

    eel.UpdateGarphData("histogram",b,g,r)()
    
# 輸入圖片更新直方圖
def histGraphByImg(img):
    
    # 單色
    if img.ndim !=3:            
        b=cv2.calcHist( [img], [0], None, [256], [0,256] )            
        b= b.astype("float").flatten().tolist()        
    # 彩色       
    else:
        b=cv2.calcHist( [img], [0], None, [256], [0,256] )            
        g=cv2.calcHist( [img], [1], None, [256], [0,256] )            
        r=cv2.calcHist( [img], [2], None, [256], [0,256] )            
        b= b.astype("float").flatten().tolist()
        g= g.astype("float").flatten().tolist()
        r= r.astype("float").flatten().tolist()    
    eel.UpdateGarphData("histogram",b,g,r)()

@eel.expose
def equalizeHist(url):
    img = data_uri_to_cv2_img(url)  
    g=img.copy()
    #單色
    if g.ndim!=3:
        g=cv2.equalizeHist( img );
    #彩色
    else:
        for i in range(0,3):
            g[:,:,i]=cv2.equalizeHist(img[:,:,i])
            
    histGraphByImg(g)
    blob = img_to_base64(g)        
    eel.SetImg(blob)()
    

# 轉灰階
@eel.expose
def toGray(url):
    img = data_uri_to_cv2_img(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blob = img_to_base64(img)
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
    ksize = float(ksize)
    img = cv2.medianBlur(img, ksize)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

# 雙邊濾波
@eel.expose
def bilateralFilter(url, _d , _sc,_ss):
    img = data_uri_to_cv2_img(url)
    d = float(_d)
    sc=float(_sc)
    ss=float(_ss)
    img = cv2.bilateralFilter(img, d,sc,ss)

    blob = img_to_base64(img)
    eel.SetImg(blob)()
    
# 銳化
@eel.expose
def unsharpMasking(url , k_size):
    img = data_uri_to_cv2_img(url)
    k_size= float(k_size)
    f_avg = cv2.GaussianBlur( img, ( 15, 15 ), 0 )
    g_mask=(img.astype(int))-(f_avg.astype(int))
    g=np.uint8(np.clip(img+k_size*g_mask,0,255))
    
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
    img2=cv2.edgePreservingFilter( img,flags=1, sigma_s=150, sigma_r=0.8)    
    
    blob = img_to_base64(img2)
    eel.SetImg(blob)()
    
# 鉛筆風
@eel.expose
def pencilStyle(url):
    img = data_uri_to_cv2_img(url)  
    img2=cv2.pencilSketch( img )
    
    blob = img_to_base64(img2)
    eel.SetImg(blob)()

# 風格化
@eel.expose
def stylization(url):
    img = data_uri_to_cv2_img(url)  
    img2=cv2.stylization( img )
    
    blob = img_to_base64(img2)
    eel.SetImg(blob)()

# 毛邊模糊
@eel.expose
def fuzzy(url):
    f = data_uri_to_cv2_img(url)  
    
    nr, nc = f.shape[:2]
    y,x = np.mgrid[0:nr,0:nc]
    x2=x+np.random.randint(-W//2,W//2,size=(nr,nc))
    y2=y+np.random.randint(-W//2,W//2,size=(nr,nc))
    x2=np.clip(x2,0,nc-1)
    y2=np.clip(y2,0,nr-1)
    g = cv2.remap( f, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR )
	
    
    blob = img_to_base64(g)
    eel.SetImg(blob)()
    
# 波紋模糊
@eel.expose
def ripple(url,amp,pid):
    img = data_uri_to_cv2_img(url)  
    amp=float(amp)
    pid=int(pid)
        
    nr, nc = img.shape[:2]
    y,x = np.mgrid[0:nr,0:nc] #沒有考慮原點位移
    x1=x+amp * np.sin( x / pid )    
    y1=y+amp * np.sin( y / pid )
    
    x2=np.clip(x1,0,nc-1)
    y2=np.clip(y1,0,nr-1)
    img2= cv2.remap( img, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR )
        
    blob = img_to_base64(img2)
    eel.SetImg(blob)()

# 放射像素風格 
@eel.expose
def radialPixelation(url, delta_r,delta_theta):
    img = data_uri_to_cv2_img(url)  
    delta_r=int(delta_r)
    delta_theta=float(delta_theta)
    
    nr, nc = img.shape[:2]
	#找到中心點座標
    y0, x0 = (nr+1) // 2, (nc+1) // 2
    # mgrid [start : end : step] 產生水平、垂直的start~end的表格
    y,x = np.mgrid[-y0:nr-y0,-x0:nc-x0]
	#算r
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
 	# 每隔delta_r 距離的像素方塊
    r2 = r - r % delta_r
	# 每隔delta_theta 角度的像素方塊
    theta2 = theta - theta % ( np.radians( delta_theta ) )
	# 極座標轉回笛卡兒座標，且位移原點回左上角
    x2=np.clip(x0+r2*np.cos(theta2),0,nc-1)
    y2=np.clip(y0+r2*np.sin(theta2),0,nr-1)
 	# remap（）圖像的重映射，可以把一幅圖像中某位置的圖元放置到另一個圖片指定位置的過程。
    img2=cv2.remap( img, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR )
    
    blob = img_to_base64(img2)
    eel.SetImg(blob)()


#放射漣漪模糊
@eel.expose
def radialRipple(url, amp,pid):
    img = data_uri_to_cv2_img(url)  
    amp=float(amp)
    pid=int(pid)
    nr, nc = img.shape[:2]
    #uv offset
    y0, x0 = (nr+1) // 2, (nc+1) // 2
    y,x = np.mgrid[-y0:nr-y0,-x0:nc-x0] #原點位移

    # 轉極座標
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)

    #波面位移
    r2 = r + np.sin(r/pid)*amp

    x2=np.clip(x0+r2*np.cos(theta),0,nc-1)
    y2=np.clip(y0+r2*np.sin(theta),0,nr-1)

    img2= cv2.remap( img, np.float32(x2), np.float32(y2), cv2.INTER_LINEAR )
    blob = img_to_base64(img2)
    eel.SetImg(blob)()


# 開啟網頁
eel.init(f'{os.path.dirname(os.path.realpath(__file__))}/web')
eel.start('main.html', mode='chrome-app')  # 網頁 (app模式)
