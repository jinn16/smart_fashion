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
function getFile(file) {
    var filepathsplit = file.split('\\');
    var filepathlength = filepathsplit.length;
    var fileName = filepathsplit[filepathlength-1]

    filePath = fileName

    return filePath;
}

