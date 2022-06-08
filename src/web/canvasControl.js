//preview 畫布拖拉
$(function () {
  $("#preview").draggable();
});

//畫布滑鼠滾輪縮放
function zoom(event) {
  event.preventDefault();

  scale += event.deltaY * -0.05;

  // Restrict scale
  scale = Math.min(Math.max(5, scale), 500);
  console.log(scale)
  console.log(preview_slider.value)
  $("#preview_slider").val(scale).change();

  // Apply scale transform
  //el.style.transform = `scale(${scale})`;
}

let scale = 100;
const el = document.querySelector("#preview");
el.onwheel = zoom;

el.addEventListener("wheel", zoom);
