<?php
include "connectToMaskDB.php";

// prepare query and parameters
$query = "UPDATE camera SET status=?, add_date=now() WHERE CameraID =?";
$status = $_GET['camera_status'];
$cameraID = $_GET['camera_id'];

if($status === "close") {
  $query = str_replace("add_date", "remove_date", $query);
}

// bind parameters and execute
$stmt = $conn->prepare($query);
$stmt->bind_param("ss", $status, $cameraID);
$stmt->execute();

$stmt->close();
$conn->close();
?>