# PhotoSharp
這是一個課堂小專題，但仍有擴充可能。
![thumb](https://i.imgur.com/DjNSDrx.png)

# 環境安裝需求

```
pip install eel
pip install numpy
pip install opencv-python
```
- haarcascade_frontalface_default.xml
- lbpcascade_frontalcatface.xml

額外.xml檔案須放置於專案根目錄。

# 啟動專案

直接執行app.py即可。
若遇到FileNotFoundError問題，請使用cmd至src資料夾下，以指令方式啟動: `python app.y`

```
FileNotFoundError: [Errno 2] No such file or directory: '....\manage.py'
```

# 輸出exe
若要輸出exe檔案需再安裝`PyInstaller`，並執行以下程式碼：
```
cd src
python -m eel app.py web
```
隨後產生build與dist資料夾，需複製影像辨識用的.xml檔案至dist資料夾下(與app.exe同層)，點擊app.exe開啟軟體。




## 新增側欄

在` <div class="sideNav_root">`tag內新增

### html

```html
<div class="opt_btns_container">
            <div class="opt_bar" data-i18n="翻譯i18n key"> 標籤名稱 </div>
            
            <div class="opt_btns close" id="標籤id">
                <!--- 範例:產生slider --->
                <script>
                    createSlider("標籤id", "input id", "label名", -359, 359, 0, 1, "onInput方法()", "onChange方法()", "label翻譯i18n key")
                </script>
                
                <!--- 範例:產生選單 --->
                <script>
                    createSelect("resize", "resize_select", "interpolation", [
                        { val: "linear", text: "雙線性", i18: "linear" },
                        { val: "nearst", text: "最近鄰", i18: "inter-nearest" },
                        { val: "cubic", text: "雙立方", i18: "inter-cubic" },
                    ]) 
                </script>
                
                <!--- 範例:產生按鈕 --->
                <script>createButton("標籤id", "OnClick方法()", "按鈕文字", "翻譯i18n key")</script>
            </div>
/div>
```

# Python端溝通API

- `data_uri_to_cv2_img(base64Str)` 接收js端傳送的based64圖片(url)並轉成python opencv可使用的圖片。

- `img_to_base64(img)` 將圖片轉成based64編碼，通常用在回傳結果圖片給前端。


範例: 將圖片轉成灰階並回傳。

### javascript

 ```javascript
 //轉灰階
function Gray() {
  b64_str = realImage;
  SaveStep(b64_str); //儲存步驟
  eel.toGray(b64_str); //呼叫python方法
}
 ```
 
### Python 
```python
# 轉灰階
@eel.expose
def toGray(url):
    img = data_uri_to_cv2_img(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blob = img_to_base64(img)
    eel.SetImg(blob)()

```

# 多語言UI擴充

編輯`i18n`資料夾下的.json檔案，在`lang.js`下load方法加入新增的json檔案位置
```javascript
$.i18n()
  .load({
    en: "/i18n/en.json",
    'zh-TW': "/i18n/zh-TW.json",
    //.....新的檔案
  })
```
