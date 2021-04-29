var img = new Image();
img.src = 'images/wallpaper.jpg';
$('#mySpinner').addClass('spinner');
var int = setInterval(function () {
	if (img.complete) {
		clearInterval(int);
		document.getElementsByTagName('body')[0].style.backgroundImage = 'url(' + img.src + ')';
		
		document.getElementById("page-container").style.display = "block";
		$('#mySpinner').removeClass('spinner');
	}
}, 50);

$('.navbar-nav .nav-link').click(function () {
	$('.navbar-nav .nav-link').removeClass('active');
	$(this).addClass('active');
})


