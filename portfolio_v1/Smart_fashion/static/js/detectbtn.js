$(function () {
    $('#detect').click(function() {
        loadingBar = document.getElementById('loadingBar')
        detect_container = $('#detect_container')
        loadingBar.style.display = 'block';
        detect_container.empty();
        $('.fourth').css("backgroundColor","#e5e5e5");

        $.ajax({
            type: 'POST',
            url: '/segajax',
            data:JSON.stringify(data),
            contentType: false,
            processData: false,
            dataType: 'json'
        }).done(function(data) {
            loadingBar.style.display = 'none';

            seg_name = data.segmentation
            folder_name = data.folder_name
            scene = data.scene
            captions = data.captions

            seg_dir = 'static/segmentation/' + folder_name + '/'

            for (var i = 0; i < scene.length; i++) {
                var cap = []
                for (var j = 0; j < captions[i].length; j++) {
                    cap += ',' + captions[i][j]
                    cap = cap.replace('undefined','').split(',').join('<br />')
                }
                detect_container.append(add(i, cap));
            }

            // 슬라이드
            var slides = document.querySelector('#detect_container'),
                slide = document.querySelectorAll('#detect_container li'),
                left=document.querySelector('.leftBtn-belows'),
                right=document.querySelector('.rightBtn-belows');
            slideWidth = 310,
                slideMargin = 15,
                currentIdex = 0,
                slideCount = slide.length + 1,
                slides.style.width = (slideWidth + slideMargin) * slideCount - slideMargin + 'px';

            function moveSlide(num) {
                slides.style.left = - num * (slideWidth + slideMargin) + 'px';
                currentIdex = num;
            }

            right.addEventListener('click', function() {
                if(currentIdex < slideCount - 3) {
                    moveSlide(currentIdex + 1);
                }else {
                    moveSlide(0);
                }
            });
            left.addEventListener('click', function() {
                if(currentIdex > 0) {
                    moveSlide(currentIdex - 1);
                }else {
                    moveSlide(slideCount - 4);
                }
            });
            $(detect_container).click(function(e){
                var id = e.target.getAttribute('id')
                $('#secVideo').replaceWith("<img id = 'secVideo' src = '" + seg_dir + seg_name[id] + ".jpg' width=\"1300px\">");
            })
            $('.fifth').css("backgroundColor","#fca311");

        })

    })
})
function add(n, cap) {
    var path = "<li><img id = \"" + n +"\" src ='" + seg_dir + seg_name[n] + ".jpg' height = \"150px\">" + `<div width = 310px>${cap}<\div>` + "</li>"
    return path
}