$(function () {
    $('#detect').click(function() {
        detect_container = $('#detect_container')
        detect_container.empty();
        $.ajax({
            type: 'POST',
            url: '/segajax',
            data:JSON.stringify(data),
            contentType: false,
            processData: false,
            dataType: 'json'
        }).done(function(data) {
            seg_name = data.segmentation
            folder_name = data.folder_name
            scene = data.scene

            seg_dir = 'static/segmentation/' + folder_name + '/'

            for (var i = 0; i < scene.length; i++) {
                detect_container.append(add(i));
            }
        })

    })
})
function add(n) {

    var path = "<li><img id = \"" + n +"\" src ='" + seg_dir + seg_name[n] + ".jpg' height = \"150px\"></li>"
    return path
}