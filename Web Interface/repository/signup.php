<?php
include "connectToCompanyDB.php";

//
if ( isset( $_POST['email'] ) && !empty( trim($_POST["email"]) ) ) {
    // // Prepare a select statement
    $emailQuery = "SELECT * FROM company_profiles WHERE Email = ?";
    $stmt = $conn->prepare($emailQuery);
    $stmt->bind_param("s", $_POST['email']);
    $stmt->execute();
    $response = $stmt->get_result(); 
    if ($response->num_rows == 1) {
        echo "Email already registered";
        $stmt->close();
        $conn->close();
        exit;
    }
    $stmt->close();
}

// prepare query and parameters
$query = "INSERT INTO company_profiles (CompanyName, Email, Password, Address, City, State, ZipCode, Phone) VALUES (?, ?, ?, ? ,?, ?, ?, ?)";

if (isset($_POST['email']) && isset($_POST['password']) && isset($_POST['companyname']) 
 && isset($_POST['address']) && isset($_POST['city']) && isset($_POST['state']) && isset($_POST['zipcode']) && isset($_POST['phone'])) {

     // bind parameters and execute
    $stmt = $conn->prepare($query);
    $password = password_hash($_POST['password'], PASSWORD_DEFAULT);
    $email = $_POST['email'];
    $stmt->bind_param("ssssssss", 
        $_POST['companyname'], 
        $email, 
        $password, 
        $_POST['address'], 
        $_POST['city'], 
        $_POST['state'], 
        $_POST['zipcode'],
        $_POST['phone']);
    $stmt->execute();
    $stmt->close();
    header('Content-Type: application/json');
    echo json_encode(['location'=>'login.html']);
}

$conn->close();
?>