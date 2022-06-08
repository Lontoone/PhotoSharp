var base64regex =
  /^([0-9a-zA-Z+/]{4})*(([0-9a-zA-Z+/]{2}==)|([0-9a-zA-Z+/]{3}=))?$/;
var realImage = "";
var isOpenHistGraph = false;
var currentFileExtension="png";

//************* EVENT ********************* */
//修改圖片事件：
$(document).on("mainImageChanged", function (e, eventInfo) {
  if (isOpenHistGraph) {
    HistGraph();
  }
  SetSidePreviewImage();
  UseChannel();
  updateFooterInfo();
});

//更新KB、size
function updateFooterInfo(){
  $("#fileSize_text").html(Math.round(realImage.length/1024)+ "KB");
  $("#resolution_text").html(preview.width + "*" + preview.height);
}


//設定src預覽圖片
try {
  eel.expose(SetImg);
} catch (ex) {}

function SetImg(newimg, updateReal = true, riseEvent = true) {
  //檢查是否為64格式:
  //console.log(newimg);
  if (base64regex.test(newimg)) {
    //preview.src = "data:image/png;base64," + newimg;
    preview.src = "data:image/"+currentFileExtension+";base64," + newimg;
  } else {
    preview.src = newimg;
  }

  if (updateReal) {
    realImage = newimg;
  }
  if (riseEvent) {
    $(document).trigger("mainImageChanged", preview);
  }
}

try {
  eel.expose(SetImgWithID);
} catch (ex) {}
function SetImgWithID(newimg, id) {
  $("#" + id).attr("src", "data:image/png;base64," + newimg);
}

function SetSidePreviewImage() {
  eel.getBlueChannels(preview.src);
}


//打開圖檔
$("#img").on("change", function () {
  var file = this.files[0];

  //blob轉base64事件
  const fr = new FileReader();
  fr.onload = function (e) {
    //更新圖片資訊
    preview.onload = function () {
      $("#resolution_text").html(preview.width + "*" + preview.height);
    };

    //preview.src = e.target.result;
    SetImg(e.target.result);
    //讀完圖片後，更新直方圖 (避免重複更新)
    if (!isOpenHistGraph) {
      HistGraph();
    }
  };

  if (file) {
    fr.readAsDataURL(file);
    console.log(file);
    currentFileExtension =file.type.split("/")[1];
    //更新 footer KB數
    $("#fileSize_text").html(Math.round(file.size / 1024) + "KB");
    //設定副檔名
    eel.setFileExtension(currentFileExtension)
  } else {
    console.log("沒有檔案");
    alert("不支援檔案類型");
  }
});

// Save as
async function SaveAs() {
  const opts = {
    types: [
      {
        description: "Images",
        accept: {
          "image/*": [".png", ".gif", ".jpeg", ".jpg"],
        },
      },
    ],
    //excludeAcceptAllOption: true,
    multiple: false,
  };

  //base64 轉 blob
  const url = decodeURIComponent(preview.src).split(",")[1];

  const imgBlob = Uint8Array.from(atob(url), (c) => c.charCodeAt(0));
  console.log(imgBlob);

  const newHandel = await window.showSaveFilePicker(opts);
  // create a FileSystemWritableFileStream to write to
  const writableStream = await newHandel.createWritable();

  // write our file
  await writableStream.write(imgBlob);

  // close the file and write the contents to disk.
  await writableStream.close();
}

// ------ side nav 點擊展開 ---------
$(".opt_btns_container .opt_bar").on("click", function () {
  var isClosed = $(this).siblings(".close").length > 0;

  var opt_btns = $(this).siblings(".opt_btns");
  if (isClosed) {
    opt_btns.removeClass("close");
  } else {
    opt_btns.addClass("close");
  }
});

// footer 預覽大小
$("#preview_slider").on("input change", function () {
  $("#preview").css("transform", "scale(" + preview_slider.value / 100 + ")");
  $("#preview_slider_text").html(preview_slider.value + "%");
});

const maxSaves = 10;
const previousSaves = [];
const nextSaves = [];
//上一步
function BackToPreviouse() {
  console.log(previousSaves.length);
  console.log(nextSaves.length);
  if (previousSaves.length > 0) {
    base_str = previousSaves.pop();
    //preview.src = base_str;
    SetImg(base_str);
    SaveNextStep(base_str);
  }
}
//下一步
function BackToNext() {
  if (nextSaves.length > 0) {
    base_str = nextSaves.pop();
    SaveStep(base_str);
    //preview.src = base_str;
    SetImg(base_str);
  }
}
//紀錄步驟
function SaveStep(bs64) {
  if (previousSaves.length < maxSaves) {
    previousSaves.push(bs64);
  } else {
    //移除第一個
    previousSaves.shift();
    previousSaves.push(bs64);
  }
}
//紀錄返回上一步步驟
function SaveNextStep(bs64) {
  if (nextSaves.length < maxSaves) {
    nextSaves.push(bs64);
  } else {
    //移除第一個
    nextSaves.shift();
    nextSaves.push(bs64);
  }
}

// ***************** 註冊 hot key *********************

function RegiseterHotKey(e) {
  console.log(e);
  //上一步
  if (e.ctrlKey && e.key === "z") {
    BackToPreviouse();
  }
  //下一步
  if (e.ctrlKey && e.key === "x") {
    BackToNext();
  }
  //存檔
  if (e.ctrlKey && e.key === "s") {
    SaveAs()
  }
}
document.addEventListener("keyup", RegiseterHotKey, false);

// ***************** --- *********************

//slider 更新數值
function updateSliderText(tid, target) {
  $(tid).html(target.value);
}

//等化器
function EqualizeHist() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  eel.equalizeHist(b64_str);
}

// 直方圖
function OpenCloseHistGraph() {
  isOpenHistGraph = !isOpenHistGraph;
  if (isOpenHistGraph) {
    HistGraph();
  }
}
async function HistGraph() {
  console.log("Update HistGraph");
  var arr = [
    106, 13, 15, 19, 31, 37, 17, 45, 24, 45, 23, 61, 78, 58, 75, 61, 98, 67,
    116, 109, 150, 165, 149, 132, 130, 135, 132, 157, 133, 124, 124, 135, 109,
    145, 176, 169, 183, 175, 137, 144, 113, 90, 85, 110, 82, 97, 94, 82, 83, 96,
    74, 93, 50, 85, 98, 107, 110, 85, 89, 96, 114, 104, 112, 178, 189, 247, 573,
    628, 800, 288, 287, 163, 152, 133, 141, 135, 152, 166, 161, 157, 158, 154,
    151, 168, 138, 137, 131, 133, 112, 134, 107, 125, 130, 132, 127, 141, 135,
    151, 128, 127, 134, 131, 136, 140, 127, 149, 113, 140, 133, 169, 162, 152,
    148, 169, 167, 157, 186, 167, 165, 151, 189, 177, 186, 168, 154, 183, 154,
    143, 157, 159, 135, 126, 132, 142, 155, 155, 148, 166, 165, 159, 157, 176,
    154, 150, 153, 169, 172, 180, 152, 138, 146, 151, 187, 165, 170, 166, 162,
    165, 172, 176, 168, 196, 160, 164, 229, 166, 221, 251, 268, 233, 227, 229,
    233, 228, 240, 254, 258, 249, 259, 256, 331, 351, 369, 299, 373, 348, 425,
    437, 394, 327, 338, 308, 321, 334, 284, 310, 373, 303, 390, 225, 193, 163,
    156, 146, 149, 131, 129, 154, 134, 147, 149, 137, 141, 146, 143, 143, 144,
    116, 143, 149, 151, 171, 165, 166, 143, 140, 138, 138, 121, 122, 149, 155,
    155, 214, 356, 361, 650, 714, 563, 1166, 842, 862, 1486, 1263, 5281, 843,
    1214, 431, 612, 328, 348, 253, 122, 198, 82, 425,
  ];

  return new Promise((resolve) => {
    setTimeout(() => {
      //b64_str = preview.src;
      b64_str = realImage;
      eel.histGraph(b64_str);
    }, 10000);
  });
  /*
  console.log("upadate hist");
  b64_str = preview.src;
  //eel.histGraph(b64_str);
  //UpdateGarphData("histogram", arr, arr, arr);
  */
}

try {
  eel.expose(UpdateGarphData);
} catch (ex) {}
function UpdateGarphData(graphID, b, g, r) {
  var labs = [Array.from({ length: 255 }).map((currentElement, i) => i)];
  /*

  const ctx = document.getElementById(graphID).getContext("2d");

  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labs[0],
      datasets: [
        {
          label: "b",
          data: b,
          backgroundColor: "blue",
        },
        {
          label: "g",
          data: g,
          backgroundColor: "green",
        },
        {
          label: "r",
          data: r,
          backgroundColor: "red",
        },
      ],
      options: {
        parsing: false,
        normalized: true,
        animation: false,
        spanGaps: true, // enable for all datasets
        showLine: false // disable for all datasets
      },
    },
  });*/

  var myChart = echarts.init(document.getElementById("histogram"), "dark", {
    renderer: "svg",
  });
  myChart.setOption({
    animation: false,
    renderer: "svg",
  });

  // 指定图表的配置项和数据
  var option = {
    title: {
      text: "",
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        label: {
          backgroundColor: "#6a7985",
        },
      },
    },
    legend: {
      data: ["R", "G", "B"],
    },
    toolbox: {
      feature: {
        saveAsImage: {},
      },
    },

    grid: {
      left: "0%",
      right: "5%",
      bottom: "1%",
      containLabel: true,
    },
    xAxis: [
      {
        type: "category",
        boundaryGap: false,
        data: labs[0],
      },
    ],
    yAxis: [
      {
        type: "value",
      },
    ],
    series: [
      {
        name: "R",
        type: "line",
        //stack: 'Total',
        areaStyle: {},
        emphasis: {
          focus: "series",
        },
        itemStyle: {
          color: "red",
        },
        data: r,
      },
      {
        name: "G",
        type: "line",
        //stack: 'Total',
        areaStyle: {},
        emphasis: {
          focus: "series",
        },
        itemStyle: {
          color: "green",
        },
        data: g,
      },
      {
        name: "B",
        type: "line",
        //stack: 'Total',
        areaStyle: {},
        emphasis: {
          focus: "series",
        },
        itemStyle: {
          color: "blue",
        },
        data: b,
      },
    ],
  };

  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
}
//-----------------濾鏡-------------------

//轉灰階
function Gray() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  eel.toGray(b64_str);
}
function NegativeFlim(){
  b64_str = realImage;
  SaveStep(b64_str);
  eel.negativeFlim(b64_str);
}
//顏色裁減
function ColorSegement(){
  b64_str = realImage;
  SaveStep(b64_str);
  var s1 = hsv_seg_mins.value
  var s2 = hsv_seg_maxs.value
  var h1 = hsv_seg_minh.value
  var h2 = hsv_seg_maxh.value
  var v1 = hsv_seg_minv.value
  var v2 = hsv_seg_maxv.value
  eel.HSV_color_segmentation(b64_str, h1,h2,s1,s2,v1,v2);
}

//轉HSV
function HSV() {
  b64_str = realImage;
  SaveStep(b64_str);
  eel.toHSV(b64_str);
}
function SetHSV_V(){
  b64_str = realImage;
  v= hsv_v.value
  SaveStep(b64_str);
  eel.setHSV_V(b64_str , v);
}
function SetHSV_H(){
  b64_str = realImage;
  v= hsv_h.value
  SaveStep(b64_str);
  eel.setHSV_H(b64_str , v);
}
function SetHSV_S(){
  b64_str = realImage;
  v= hsv_s.value
  SaveStep(b64_str);
  eel.setHSV_S(b64_str , v);
}

//曝光
function Gamma(){
  b64_str = realImage;
  v= gammaCorrection.value
  SaveStep(b64_str);
  eel.gamma(b64_str , v);
}
//對比
function Beta(){
  b64_str = realImage;
  a= betaCorrection_a.value
  b= betaCorrection_b.value
  SaveStep(b64_str);
  eel.beta(b64_str , a,b);
}


//高斯模糊
function GaussianBlur() {
  //console.log("gaussian" + realImage);
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  kSize = gs_k_size.value * 2 - 1;
  sigma = gs_sigma_size.value;
  eel.gaussianblur(b64_str, kSize, sigma);
}

//中值濾波
function MedianFilter() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  kSize = mf_k_size.value;
  eel.mddianFilter(b64_str, kSize);
}
//雙邊濾波
function BilateralFilter() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  dSize = bf_d_size.value;
  scSize = bf_cs_size.value;
  ssSize = bf_ss_size.value;
  eel.bilateralFilter(b64_str, dSize, scSize, ssSize);
}

//非銳化遮罩
function USmasking() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  kSize = usm_k_size.value;
  eel.unsharpMasking(b64_str, kSize);
}

//Resize
function Resize() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  ratio = resize_ratio.value;
  select = resize_select.value;
  eel.resize(b64_str, ratio, select);  
}
function HorizontalFlip(){
  b64_str = realImage;
  SaveStep(b64_str);
  eel.horizontalFlip(b64_str);  

}
function VerticalFlip(){
  b64_str = realImage;
  SaveStep(b64_str);
  eel.verticalFlip(b64_str); 
}

function Rotate(){
  angle = rotateAngle.value;
  /*
  originCss= $('#preview').css('transform');
  if(originCss=="none"){
    originCss =""
  }
  newCss = originCss + 'rotate(' + angle + 'deg)';
  console.log(newCss)
  $('#preview').css('transform',newCss);*/
  $('#preview').css('transform','rotate(' + angle + 'deg')

}
function ConfirmRotation(){
  //取消預覽效果
  $('#preview').css('transform','rotate(' + 0 + 'deg')
  //真的寫成圖片
  b64_str = realImage;
  angle = rotateAngle.value;
  eel.rotate(b64_str,angle);
}

//----------------效果--------------------
//邊緣提取
function Grad() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  k_type = grad_select.value;
  eel.grad(b64_str, k_type);
}


//邊緣保留
function EdgePreserve() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  eel.edgePreserve(b64_str);
}

//鉛筆風
function PencilStyle() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  eel.pencilStyle(b64_str);
}

//風格化
function Stylization() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  eel.stylization(b64_str);
}

//毛邊模糊
function Fuzzy() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  eel.fuzzy(b64_str);
}

// 波紋模糊
function Ripple() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  amp = ripple_amp.value;
  pid = ripple_pid.value;

  eel.ripple(b64_str, amp, pid);
}

// 放射像素風格
function RadialPixelation() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  r = rp_r.value;
  theta = rp_theta.value;

  eel.radialPixelation(b64_str, r, theta);
}

// 漣漪效果
function RadialRipple() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  amp = radical_ripple_amp.value;
  pid = radical_ripple_pid.value;

  eel.radialRipple(b64_str, amp, pid);
}

//魚眼效果
function Fisheye() {
  //b64_str = preview.src;
  b64_str = realImage;
  SaveStep(b64_str);
  eel.fisheye(b64_str);
}

//Twirl Effect
function Twirl() {
  //b64_str = preview.src;
  b64_str = realImage;
  k = twirl_k.value;
  SaveStep(b64_str);
  eel.twirl(b64_str, k);
}

// 運動模糊
function MotionBlur() {
  //b64_str = preview.src;
  b64_str = realImage;
  l = motionBlur_l.value;
  a = motionBlur_a.value;
  SaveStep(b64_str);
  eel.motionBlur(b64_str, l, a);
}

//放射模糊
function RadicalBlur() {
  //b64_str = preview.src;
  b64_str = realImage;
  k = radical_blur_size.value;
  SaveStep(b64_str);
  eel.radicalBlur(b64_str, k);
}

//人臉辨識
function FaceDetection() {
  //b64_str = preview.src;
  b64_str = realImage;
  algorithmSelect= face_dec_select.value;
  scale = face_dec_scale.value;
  neighbor = face_dec_neighbor.value;
  SaveStep(b64_str);
  eel.faceDetection(b64_str, scale, neighbor,algorithmSelect);
}


//疊圖層
function OverlayerImage(){
  b64_str = realImage;
  layer_b64_str =newLayerImage.src ;
  opt=overlayMethod.value;
  //x=offset_x.value;
  //y=offset_y.value;
  SaveStep(b64_str);
  eel.overlayImage(b64_str,layer_b64_str,opt)

}