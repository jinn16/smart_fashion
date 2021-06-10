// file = $('#file')
//
// $(function() {
//     file.bind({
//         click: function () {
//             document.getElementById('loadingBar').style.display = 'block';
//             $('#secVideo').replaceWith("<video autoplay = autoplay id = 'secVideo' src = ''>");
//         },
//         change: function() {
//             $('.container').css("backgroundColor","#e5e5e5");
//             $('.second').css("backgroundColor","#fca311");
//             data = document.querySelector('#file').value
//             document.getElementById('loadingBar').style.display = 'none';
//         }
//     })
// })
//
$(function() {
    $('#file').click(function () {
        document.getElementById('loadingBar').style.display = 'block';
        $('#secVideo').replaceWith("<video autoplay = autoplay id = 'secVideo' src = ''>");
    }

    )
    $('#file').change(function(submitFile){
        $('.container').css("backgroundColor","#e5e5e5");
        $('.second').css("backgroundColor","#fca311");
        data = document.querySelector('#file').value
        // outputfile.value = getFile(data);
        document.getElementById('loadingBar').style.display = 'none';
    })
})