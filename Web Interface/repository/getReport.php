<?php
include "connectToMaskDB.php";

// retrieve/initiate params
$data = array();
$selectedWeek = $_GET['selectedWeek'];
$selectedYear = $_GET['selectedYear'];

// open first statement: get overall stats of specified week 
$weeklyEmpNoMaskAvgsql = "SELECT 
CONCAT('Week ', WEEK(capture_date)) AS 'week_nbr', 
(SELECT COUNT(*) FROM results WHERE WEEK(capture_date)=? AND YEAR(capture_date)=?) AS 'total_result',
(SELECT COUNT(*) FROM employee) AS 'total_employee', 
COUNT(*) AS 'total_result_employee_occurance', 
COUNT(DISTINCT(A.employeeId)) AS 'total_result_employee'
FROM results_employee A INNER JOIN employee B ON A.employeeId = B.employeeId INNER JOIN results C ON C.ResultsID = A.ResultsID 
WHERE WEEK(capture_date)=? and YEAR(capture_date)=?
GROUP BY WEEK(capture_date)";

// bind parameters and execute
$stmt = $conn->prepare($weeklyEmpNoMaskAvgsql);
$stmt->bind_param("iiii", $selectedWeek, $selectedYear, $selectedWeek, $selectedYear);
$stmt->execute();

// get the mysqli result
$response = $stmt->get_result(); 

if ($response->num_rows > 0) {
    while ($row = $response->fetch_assoc()) {
        $data['overallStats'] = $row;
    }
}

// close first statement
$stmt->close();


// open second statement: gets details of all employees that didn't wear mask that week
$empNoMaskSql = "SELECT 
B.EmployeeID,
CONCAT(TRIM(B.First_Name), ' ', TRIM(B.Last_Name)) AS 'Employee_Name',
B.Title,
B.Email,
B.Phone, 
capture_date,
capture_time, 
camera.location
FROM results_employee A INNER JOIN employee B ON A.employeeId = B.employeeId INNER JOIN results C ON C.ResultsID = A.ResultsID INNER JOIN camera ON camera.CameraID = C.CameraID
WHERE WEEK(capture_date)=? and YEAR(capture_date)=?
ORDER BY B.EmployeeID";

$stmt = $conn->prepare($empNoMaskSql);
$stmt->bind_param("ii", $selectedWeek, $selectedYear);
$stmt->execute();
$response = $stmt->get_result(); 

if ($response->num_rows > 0) {
    while ($row = $response->fetch_assoc()) {
        $data['employeeDetails'][] = $row;
    }
}

$stmt->close();

// return data
if (!empty($data)) {
    echo json_encode($data);
}

$conn->close();
?>