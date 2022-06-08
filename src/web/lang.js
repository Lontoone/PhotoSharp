$.i18n({
  //locale: "zh-TW", // Locale is Hebrew
  //locale: "en", // Locale is Hebrew
});

$.i18n()
  .load({
    en: "/i18n/en.json",
    'zh-TW': "/i18n/zh-TW.json",
  })
  .done(function () {
    $('body').i18n();
  });

$("title").i18n("appname-title");

function SetLanguage(lang){
  console.log("switch lange "+lang)
  $.i18n().locale = lang;
  $('body').i18n();
}