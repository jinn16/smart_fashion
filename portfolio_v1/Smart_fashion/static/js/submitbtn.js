$(function submitFile() {
    $('#submit').click(function() {

        document.getElementById('loadingBar').style.display = 'block';
        $('.lists>div').animate({opacity:"1"},100);
        $('.container').css("backgroundColor","#e5e5e5");
        $('.third').css("backgroundColor","#fca311");


        $("div#images_container").empty();

        event.preventDefault();
        var form_data = new FormData($('#upload')[0]);
        $.ajax({
            type: 'POST',
            url: '/uploadajax',
            data: form_data,
            contentType: false,
            processData: false,
            dataType: 'json',

        }).done(function(data) {
            document.getElementById('loadingBar').style.display = 'none';

            file = data.name
            scene = data.scene
            video = data.video
            file_name = file.replace(/(.png|.jpg|.jpeg|.gif|.mp4|.mp3|.ogg|.avi|.mov)$/, '');
            scene_dir = '../static/scene/' + file_name + '/'
            start_time = data.start
            end_time = data.end
            scene_num = data.scene_num
             // frames 추가 by hsy
            frames = data.frame
            console.log(start_time, end_time, frames)

            for (var i = 0; i < scene.length; i++) {
                $("div#images_container").append(add_image(i));
            }

            // $('.lists>div').animate({opacity:"0"},100);
            $('.lists>div').css("display","none");

            var slides = document.querySelector('#images_container'),
                slide = document.querySelectorAll('#images_container li'),
                currentIdex = 0,
                slideCount = slide.length+1,
                left=document.querySelector('.leftBtn'),
                right=document.querySelector('.rightBtn');
                slideWidth = 310,
                slideMargin = 15,

                slides.style.width = (slideWidth + slideMargin)*slideCount -slideMargin +'px';
            function moveSlide(num) {
                slides.style.left = -num * 330 + 'px';
                currentIdex = num;
            }
            right.addEventListener('click', function() {
                if(currentIdex<slideCount-3) {

                    moveSlide(currentIdex+1);
                }else {
                    moveSlide(0);
                }
            });

            left.addEventListener('click', function() {
                if(currentIdex>0) {
                    moveSlide(currentIdex-1);
                }else {
                    moveSlide(slideCount-4);
                }
            });

            $('div#images_container').click(function(e) {
                var id = e.target.getAttribute('id')
                $('#secVideo').replaceWith("<video autoplay = autoplay id = 'secVideo' src = '" + scene_dir + video[id] + "' controls width=\"1300px\">");
                console.log(start_time[id], '-', end_time[id])
            })

            $('div#images_container').mouseover(function(e) {
                var id = e.target.getAttribute('id')
                document.getElementById(id).setAttribute('title', start_time[id]+ '-' + end_time[id]);
            })

            $('.container').css("backgroundColor","#e5e5e5");
            $('.fourth').css("backgroundColor","#fca311");
    

        }).fail(function(data){
            alert('error!');
        });
    });
});

function add_image(i) {
    var path = "<li><img id = \"" + i +"\" src ='" + scene_dir + scene[i] + "' height = \"150px\" title = \"hello\"></li>"
    return path
}

