$(document).ready(function() {

});

var url_query;

function searchByQuery(){
    query = $('#query').val();
    url_query = '/do_ranked_search/' + query;
    $('#query').val('')

    var e = '';
    e = e + '<tr><th>ID</th><th>Score</th></tr><th>Text</th></tr>'

    $.getJSON(url_query,function(data){
    var i = 0;
    $.each(data, function(){
        e = e + '<tr>'
        e = e + '<td>'+data[i]['name']+'</td>'

        e = e + '<td>'+data[i]['weight']+'</td>'
        e = e + '<td>'+data[i]['text']+'</td>'

        e = e + '</tr>'
        i = i+1;
    });
    $('#boxResults').html(e);
    });
}