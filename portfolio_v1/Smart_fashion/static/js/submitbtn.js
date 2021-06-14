$(function submitFile() {
    $('#submit').click(function() {
        // 변수 및 id 값 선언
        loadingBar = document.getElementById('loadingBar')
        images_container = $('#images_container')
        container = $('.container')
        list = $('.lists>div')
        var form_data = new FormData($('#upload')[0]);

        loadingBar.style.display = 'block';
        images_container.empty();
        event.preventDefault();

        list.animate({opacity:"1"},100);
        container.css("backgroundColor","#e5e5e5");
        $('.third').css("backgroundColor","#fca311");

        $.ajax({
            type: 'POST',
            url: '/uploadajax',
            data: form_data,
            contentType: false,
            processData: false,
            dataType: 'json',
            success: function (data) {
                $('.items').html(data);
            }
        }).done(function(data) {

            loadingBar.style.display = 'none';

            // json 파일 불러오기
            file = data.dir_name
            scene = data.scene
            video = data.video
            start_time = data.start
            end_time = data.end
            frames = data.frame

            scene_dir = file + '/'

            // image container에 사진 추가
            for (var i = 0; i < scene.length; i++) {
                const items = document.querySelector('.items')

                images_container.append(add_image(i));

                const item = creatLines(i);
                items.appendChild(item);
                item.scrollIntoView({block:'center'});

            }

            list.css("display","none");

            // 슬라이드
            var slides = document.querySelector('#images_container'),
                slide = document.querySelectorAll('#images_container li'),
                left=document.querySelector('.leftBtn'),
                right=document.querySelector('.rightBtn');
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

            // image container 관련 이벤트
            $(function() {
                images_container.bind({
                    click: function(e) {
                        var id = e.target.getAttribute('id')
                        const item = creatLines(id);
                        viewVideo(id);
                        item.scrollIntoView({block:'center'});
                    },
                    mouseover: function(e) {
                        var id = e.target.getAttribute('id')
                        document.getElementById(id).setAttribute('title', `Scene Number: ${id}` + `\n`+  start_time[id]+ '-' + end_time[id]);
                    }
                })
            })

            const items = document.querySelector('.items');
            items.addEventListener('click', event=> {
                var el = event.target.dataset.id
                if(el){
                    viewVideo(el);
                    moveSlide(el);

                }
            });

            container.css("backgroundColor","#e5e5e5");
            $('.fourth').css("backgroundColor","#fca311");

        }).fail(function(data){
            alert('error!');
        });
    });
});

function add_image(n) {

    var path = "<li><img id = \"" + n +"\" src ='" + scene_dir + scene[n] + "' height = \"150px\" title = \"hello\">" + `<div>scene number: ${n}<br>${start_time[n]} - ${end_time[n]}<\div>` + "</li>"
    return path
}

function creatLines(n) {
    const itemRow = document.createElement('li');
    itemRow.setAttribute('class','item__row');
    itemRow.innerHTML=`
        <div class="item" data-id=${n}>
              <span class="item__name"><p>${n}</p></span>
              <span class="item__frame"><p>${frames[n]}</p></span>
              <span class="item__time"><p>${start_time[n]}</p></span>
        </div>
        <div class="item__divider"></div>`
    return itemRow;
}

function viewVideo(n) {
    $('#secVideo').replaceWith("<video autoplay = autoplay id = 'secVideo' src = '" + scene_dir + video[n] + "' controls width=\"1300px\">");
}