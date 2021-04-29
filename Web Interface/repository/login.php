<?php

include "connectToCompanyDB.php";
$error_msg = "Incorrect username or password";

if ( isset( $_POST['userName'] ) && isset( $_POST['password'] ) ) {
    if ( !empty(trim($_POST["userName"])) && !empty(trim($_POST["password"])) ) {
        // Getting submitted user data from database
        $query = "SELECT * FROM company_profiles WHERE Email = ?";
        $stmt = $conn->prepare($query);
        $stmt->bind_param('s', $_POST['userName']);
        $stmt->execute();
        $result = $stmt->get_result();
    	$user = $result->fetch_object();
    		
    	// Verify user password and set $_SESSION
    	if ( password_verify( $_POST['password'], $user->Password ) ) {
    		// $_SESSION['user_id'] = $user->ID;
            header('Content-Type: application/json');
            echo json_encode(['location'=>'admin_panel.html']);
    	} else {
            echo $error_msg;
        }
    } else {
        echo $error_msg;
    }
}

$conn->close();
?>