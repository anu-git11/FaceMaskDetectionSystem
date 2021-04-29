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

// login form listener
$(function () {
    $('button[id="login-form-submit"]').on("click", function () {
        let userName = document.getElementById("userName").value;
        let password = document.getElementById("password").value;

        if (userName && password) {
            $.ajax({
                method: "POST",
                url: "repository/login.php",
                data: { 
                    "userName": userName,
                    "password": password
                }
            }).done(function (response) {
                if (response) {
                    if (response.location) {
                        window.location.href = response.location;
                    } else {
                        $('input').addClass('invalid');
                        document.getElementById("error").innerHTML = "Please fill out all required fields.";
                        document.getElementById("error").innerHTML = `<i class="fas fa-exclamation-triangle"></i>&nbsp;&nbsp;` + response;
                    }
                }
            });
        }
    });
});

// register 
var errorIcon = '<i class="fas fa-exclamation-triangle"></i>&nbsp;&nbsp;';

/*
* displays error message and mark input as invalid
*/
function displayMessage(controlField, id, errorMsg = '', addClass = false) {
    if (addClass) {
        $(controlField).addClass('invalid');
        document.getElementById(id).innerHTML = `${errorIcon}${errorMsg}`;
    } else {
        $(controlField).removeClass('invalid');
        document.getElementById(id).innerHTML = ``;
    }
}

/*
* checks input value if 1) has value 2) has correct format
*/
function checkInputValue(inputControl) {
    if (inputControl.value) {
        if(inputControl.id === 'email') {
            // validate email format
            let mailformat = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;
            if (!inputControl.value.match(mailformat)) {
                displayMessage(`input[id="${inputControl.id}"]`, 'email-error-msg', 'Must be an email address', true);
                // $(`input[id="${inputControl.id}"]`).addClass('invalid');
                // document.getElementById("email-error-msg").innerHTML = `${errorIcon}Must be an email address`;
            } else {
                displayMessage(`input[id="${inputControl.id}"]`, 'email-error-msg');
                // $(`input[id="${inputControl.id}"]`).removeClass('invalid');
                // document.getElementById("email-error-msg").innerHTML = ``;
            }
        } else if (inputControl.id === 'phone') {
            let phoneno = /^\d{10}$/;

            if (!inputControl.value.match(phoneno)) {
                displayMessage(`input[id="${inputControl.id}"]`, 'phone-error-msg', 'Must be exactly 10 digit numbers', true);
                // $(`input[id="${inputControl.id}"]`).addClass('invalid');
                // document.getElementById("phone-error-msg").innerHTML = `${errorIcon}Must be exactly 10 digit numbers`;
            } else {
                displayMessage(`input[id="${inputControl.id}"]`, 'phone-error-msg');
                // $(`input[id="${inputControl.id}"]`).removeClass('invalid');
                // document.getElementById("phone-error-msg").innerHTML = ``;
            }
        } else if (inputControl.id === 'zipcode') {
            let zipcodeFormat = /^\d+$/;
            let allDigits = inputControl.value.match(zipcodeFormat);
            let withinMaxChar = inputControl.value.length <= 10;

            if (!allDigits || !withinMaxChar) {
                displayMessage(`input[id="${inputControl.id}"]`, 'zipcode-error-msg',
                    !allDigits ? `Must be numbers only` : `Maximum 10 characters only`, true);
                // $(`input[id="${inputControl.id}"]`).addClass('invalid');
                // document.getElementById("zipcode-error-msg").innerHTML = !allDigits ? `${errorIcon}Must be numbers only` : `${errorIcon}Maximum 10 characters only`;
            } else{
                displayMessage(`input[id="${inputControl.id}"]`, 'zipcode-error-msg');
                // $(`input[id="${inputControl.id}"]`).removeClass('invalid');
                // document.getElementById("zipcode-error-msg").innerHTML = ``;
            }
        } else {
            $(`input[id="${inputControl.id}"]`).removeClass('invalid');
        }
    } else {
        $(`input[id="${inputControl.id}"]`).addClass('invalid');
    }
}

/*
* checks whether the 2 password matched
*/
function checkPasswordMatch() {
    let password = document.getElementById("password").value;
    let confirmpassword = document.getElementById("confirmpassword").value;

    if (password && confirmpassword && (password !== confirmpassword)) {
        displayMessage('input[id="confirmpassword"]', 'confirmpassword-error-msg', 'Passwords must match', true);
        // $('input[id="confirmpassword"]').addClass('invalid');
        // document.getElementById("confirmpassword-error-msg").innerHTML = `${errorIcon}Passwords must match`;
    } else {
        displayMessage('input[id="confirmpassword"]', 'confirmpassword-error-msg');
        // $('input[id="confirmpassword"]').removeClass('invalid');
        // document.getElementById("confirmpassword-error-msg").innerHTML = ``;
    }
}

$(function () {
    // check matching passwords
    $('input[id="password"]').on("input", checkPasswordMatch);
    $('input[id="confirmpassword"]').on("input", checkPasswordMatch);

    // on form submit
    $('button[id="signup-form-submit"]').on("click", function () {
		console.log("abc")
        // check all inputs has value in it
        $('#signup-form').find('select, input').each(function () {
            // if input doesn't have value, display error msg
            if (!$(this).val()) {
                displayMessage(this, 'required-error-msg', 'Please fill out all fields', true);
                // $(this).addClass('invalid');
                // document.getElementById("required-error-msg").innerHTML = `${errorIcon}Please fill out all fields`;
            }
        });

        // if form is valid
        if (!$('input').hasClass('invalid')) {
			console.log("kkk")
            $.ajax({
                method: "POST",
                url: "repository/signup.php",
                data: {
                    email: $("#email").val(),
                    password: $("#password").val(),
                    companyname: $("#companyname").val(),
                    address: $("#address").val(),
					city: $("#city").val(),
                    state: $("#state").val(),
                    zipcode: $("#zipcode").val(),
                    phone: $("#phone").val(),
                }
            }).done(function (response) {
				console.log(response)
                if (response) {
                    // if inserted, redirect to login page; else display error msg
                    if (response.location) {
                        window.location.href = response.location;
                    } else {
                        document.getElementById("required-error-msg").innerHTML = '';
                        displayMessage('input[id="email"]', 'email-error-msg', response, true);
                        // $('input[id="email"]').addClass('invalid');
                        // document.getElementById("email-error-msg").innerHTML = `${errorIcon}${response}`;
                    }
                }
            }).fail( function(xhr, textStatus, errorThrown) {
        alert(xhr.responseText);
    });
        }
    });
});

