function LoadLayerImage(target) {
  console.log(target);
}

//打開圖檔
$("#layerImgInput").on("change", function () {
  var file = this.files[0];
  console.log(file);

  //blob轉base64事件
  const fr = new FileReader();
  fr.onload = function (e) {
    newLayerImage.src = e.target.result;
    console.log(e);
  };

  if (file) {
    fr.readAsDataURL(file);    

  } else {
    console.log("沒有檔案");
    alert("不支援檔案類型");
  }
});

