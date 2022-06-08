function toggle(target){    
    var isOpen=$(target).attr("isToggled")
    
    if(isOpen==="true"){
        $(target).attr("isToggled","false")
        $(target).attr("src","./img/eye_slash.png")
    }
    else{
        $(target).attr("isToggled","true")
        $(target).attr("src","./img/eye.png")
    }

    UseChannel();
}

async function UseChannel(){
    var _open_r=$("#ct_r").attr("isToggled")==="true"
    var _open_g=$("#ct_g").attr("isToggled")==="true"
    var _open_b=$("#ct_b").attr("isToggled")==="true"

    console.log(_open_b);
    eel.useChannel(realImage, _open_r,_open_g,_open_b);
}