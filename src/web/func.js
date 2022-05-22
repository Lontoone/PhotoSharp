//設定src預覽圖片
eel.expose(SetImg);

function SetImg(newimg) {
  console.log(newimg);
  preview.src = "data:image/png;base64," + newimg;
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

    preview.src = e.target.result;
    currentFile = e.target.result;
  };

  if (file) {
    fr.readAsDataURL(file);
    //更新 footer KB數
    $("#fileSize_text").html(Math.round(file.size / 1024) + "KB");
  } else {
    console.log("沒有檔案");
  }
  console.log(file);
});

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

const maxSaves=10;
const previousSaves = [];
const nextSaves = [];
//上一步
function BackToPreviouse() {
  console.log(previousSaves.length);
  console.log(nextSaves.length);
  if (previousSaves.length > 0) {
    base_str=previousSaves.pop();
    preview.src = base_str;
    SaveNextStep(base_str)
  }
}
//下一步
function BackToNext(){
  if (nextSaves.length > 0) {
    base_str=nextSaves.pop();
    SaveStep(base_str);
    preview.src = base_str;
  }
}
//紀錄步驟
function SaveStep(bs64){  
  if(previousSaves.length<maxSaves){
    previousSaves.push(bs64);
  }
  else{
    //移除第一個
    previousSaves.shift();
    previousSaves.push(bs64);
  }
}
//紀錄返回上一步步驟
function SaveNextStep(bs64){
  if(nextSaves.length<maxSaves){
    nextSaves.push(bs64);
  }
  else{
    //移除第一個
    nextSaves.shift();
    nextSaves.push(bs64);
  }
}

// ***************** 註冊 hot key *********************


function RegiseterHotKey(e) {
  console.log(e);  
  //上一步
  if (e.ctrlKey && e.key === 'z') {      
    BackToPreviouse();
  }
  //下一步
  if (e.ctrlKey && e.key === 'x') {      
    BackToNext();
  }
}
document.addEventListener('keyup', RegiseterHotKey, false);

// ***************** --- *********************

//slider 更新數值
function updateSliderText(tid, target) {
  $(tid).html(target.value);
}

//等化器
function EqualizeHist() {
  b64_str = preview.src;
  SaveStep(b64_str);
  eel.equalizeHist(b64_str);
}

// 直方圖
function HistGraph() {
  //var arr =[106,13,15,19,31,37,17,45,24,45,23,61,78,58,75,61,98,67,116,109,150,165,149,132,130,135,132,157,133,124,124,135,109,145,176,169,183,175,137,144,113,90,85,110,82,97,94,82,83,96,74,93,50,85,98,107,110,85,89,96,114,104,112,178,189,247,573,628,800,288,287,163,152,133,141,135,152,166,161,157,158,154,151,168,138,137,131,133,112,134,107,125,130,132,127,141,135,151,128,127,134,131,136,140,127,149,113,140,133,169,162,152,148,169,167,157,186,167,165,151,189,177,186,168,154,183,154,143,157,159,135,126,132,142,155,155,148,166,165,159,157,176,154,150,153,169,172,180,152,138,146,151,187,165,170,166,162,165,172,176,168,196,160,164,229,166,221,251,268,233,227,229,233,228,240,254,258,249,259,256,331,351,369,299,373,348,425,437,394,327,338,308,321,334,284,310,373,303,390,225,193,163,156,146,149,131,129,154,134,147,149,137,141,146,143,143,144,116,143,149,151,171,165,166,143,140,138,138,121,122,149,155,155,214,356,361,650,714,563,1166,842,862,1486,1263,5281,843,1214,431,612,328,348,253,122,198,82,425];
  b64_str = preview.src;
  SaveStep(b64_str);
  eel.histGraph(b64_str);
  //UpdateGarphData("histogram",arr);
}

eel.expose(UpdateGarphData);
function UpdateGarphData(graphID, b, g, r) {
  var labs = [Array.from({ length: 255 }).map((currentElement, i) => i)];

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
    },
  });
}
//-----------------濾鏡-------------------

//轉灰階
function Gray() {
  b64_str = preview.src;
  SaveStep(b64_str);
  eel.toGray(b64_str);
}

//高斯模糊
function GaussianBlur() {
  b64_str = preview.src;
  SaveStep(b64_str);
  kSize = gs_k_size.value * 2 - 1;
  sigma = gs_sigma_size.value;
  eel.gaussianblur(b64_str, kSize, sigma);
}

//中值濾波
function MedianFilter() {
  b64_str = preview.src;
  SaveStep(b64_str);
  kSize = mf_k_size.value;
  eel.gaussianblur(b64_str, kSize);
}
//雙邊濾波
function BilateralFilter() {
  b64_str = preview.src;
  SaveStep(b64_str);
  dSize = bf_d_size.value;
  scSize = bf_cs_size.value;
  ssSize = bf_ss_size.value;
  eel.gaussianblur(b64_str, dSize, scSize, ssSize);
}

//非銳化遮罩
function USmasking() {
  b64_str = preview.src;
  SaveStep(b64_str);
  kSize = usm_k_size.value;
  eel.unsharpMasking(b64_str, kSize);
}

//Resize
function Resize() {
  b64_str = preview.src;
  SaveStep(b64_str);
  ratio = resize_ratio.value;
  select = resize_select.value;

  eel.resize(b64_str, ratio, select);
  console.log(kSize);
}

//----------------效果--------------------
//邊緣提取
function Grad() {
  b64_str = preview.src;
  SaveStep(b64_str);
  k_type = grad_select.value;
  eel.grad(b64_str, k_type);
}

//邊緣保留
function EdgePreserve() {
  b64_str = preview.src;
  SaveStep(b64_str);
  eel.edgePreserve(b64_str);
}

//鉛筆風
function PencilStyle() {
  b64_str = preview.src;
  SaveStep(b64_str);
  eel.pencilStyle(b64_str);
}

//風格化
function Stylization() {
  b64_str = preview.src;
  SaveStep(b64_str);
  eel.stylization(b64_str);
}

//毛邊模糊
function Fuzzy() {
  b64_str = preview.src;
  SaveStep(b64_str);
  eel.fuzzy(b64_str);
}

// 波紋模糊
function Ripple() {
  b64_str = preview.src;
  SaveStep(b64_str);
  amp = ripple_amp.value;
  pid = ripple_pid.value;

  eel.ripple(b64_str, amp, pid);
}

// 放射像素風格
function RadialPixelation() {
  b64_str = preview.src;
  SaveStep(b64_str);
  r = rp_r.value;
  theta = rp_theta.value;

  eel.radialPixelation(b64_str, r, theta);
}

// 漣漪效果
function RadialRipple() {
  b64_str = preview.src;
  SaveStep(b64_str);
  amp = radical_ripple_amp.value;
  pid = radical_ripple_pid.value;

  eel.radialRipple(b64_str, amp, pid);
}
