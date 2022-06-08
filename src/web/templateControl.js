function guidGenerator() {
    var S4 = function() {
       return (((1+Math.random())*0x10000)|0).toString(16).substring(1);
    };
    return (S4()+S4()+"-"+S4()+"-"+S4()+"-"+S4()+"-"+S4()+S4()+S4());
}

function createButton(parent,onClick,text,i18=""){
    var btn = $("#temp_side_nav_button").html();
    var new_btn= $(btn).clone();
    new_btn.html(text)
    var parent = $("#"+parent);
    
    $(new_btn).attr('onClick',onClick)
    $(new_btn).attr('data-i18n',i18)
    parent.append(new_btn);
}


function createSlider(parent,sliderID,labelText="label",min,max,_default,step=1 ,onInput="",onChange="",i18=""){
    var btn = $("#temp_side_nav_slider").html();
    var parent = $("#"+parent);

    var new_btnGroup= $(btn).clone();
    
    var label_container= $($(new_btnGroup)[0]).find("#opt-btn-label-container")
    var label= $($(label_container)[0]).find("#tmp_labelText")
    var labelValue= $($(label_container)[0]).find("#tmp_label_value_text")
    var input= $($(new_btnGroup)[0]).find("#tmp_slider")

    label.html(labelText)
    labelValue.html(_default)
    //自動調整label文字
    $(input).on("input",()=>{
        text = $(input).val();
        labelValue.html(text)
    })
    $(label).attr("id", ("sp"+i18))
    $(label).attr("data-i18n", i18)
    //label.append(labelValue)
    //slider 參數
    $(input).attr('min',min)
    $(input).attr('max',max)
    $(input).attr('value',_default)    
    $(input).attr('step',step)
    $(input).attr('id',sliderID)
    $(input).attr('OnInput',onInput)
    $(input).attr('OnChange',onChange)
    
    
    parent.append(new_btnGroup);
}


function createSelect(parent,selectID,labelText,opts=[{val:"temp", text:"temp",i18:""}]){
    var container = $("#temp_side_nav_select").html();
    var opt=  $("#temp_side_nav_opt").html();
    var parent = $("#"+parent);

    var new_btnGroup= $(container).clone();
    var label= $($(new_btnGroup)[0]).find("#tmp_labelText")
    var select= $($(new_btnGroup)[0]).find("#tmp_selectid")

    var label_id= guidGenerator();
    $(select).attr("name",label_id)
    $(select).attr("id",selectID)
    $(label).attr("data-i18n",labelText)
    $(label).attr("for",label_id)
    label.html(labelText)

    for(var i =0;i<opts.length;i++){
        var _opt=$(opt).clone();
        $(_opt).attr("value",opts[i].val)
        $(_opt).attr("data-i18n",opts[i].i18)
        _opt.html(opts[i].text)
        $(select).append(_opt)
    }
    

    parent.append(new_btnGroup);

}