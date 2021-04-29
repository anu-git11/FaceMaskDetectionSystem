<?php
include "connectToMaskDB.php";

$sql = "SELECT * FROM camera";

$result = $conn->query($sql);

if ($result->num_rows > 0) {
  // output result array
  $response = array();
  while($row = $result->fetch_assoc()) {
    $response[] = $row;
  }
  echo json_encode($response);
}

$conn->close();
?>