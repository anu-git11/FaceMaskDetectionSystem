function toggleCamera(camID) {
    let currentStatus = document.getElementById(`myonoffswitch_${camID}`).checked;
    let statusToChange = currentStatus ? "close" : "open";
    $.ajax({
        url: "repository/toggleCamera.php",
        data: { "camera_status": statusToChange, "camera_id": camID }
    }).done(
        loadCamera
    );
}

function loadCamera() {
    $.ajax({
		url: "repository/getCameraList.php"
    }).done(function (data) {
        data = JSON.parse(data)

        let tableData = "";
        for (var i = 0; i < data.length; i++) {
            // pre-define initial checked value
            isCamOpen = data[i]['Status'] == "open" ? "checked" : "";
            tableData += `
				<tr>
					<td> 
						<div id="CameraID"> ${data[i]['CameraID']} </div> 
					</td>
					<td> 
						<div id="Location"> ${data[i]['Location']} </div> 
					</td>
					<td> 
						<div id="AddDate"> ${data[i]['Add_Date']} </div>
					</td>
					<td> 
						<div id="RemoveDate"> ${data[i]['Remove_Date']} </div>
					</td>
                    <td>
						<div id="Status"> ${data[i]['Status']} </div>
					</td>
					<td> 
						<div class="onoffswitch">
							<input type="checkbox" name="onoffswitch" class="onoffswitch-checkbox" id="myonoffswitch_${data[i]['CameraID']}" tabindex="0" ${isCamOpen}>
							<label class="onoffswitch-label" for="myonoffswitch_${data[i]['CameraID']}" onclick="toggleCamera('${data[i]['CameraID']}');">
								<span class="onoffswitch-inner"></span>
								<span class="onoffswitch-switch"></span>
							</label>
						</div>
					</td>
				</tr>
			`;
			
        }

        let pageContent = `
			<h1 class="text-center"> CAMERA </h1>
            <div class="table-container">
			<div class="table-responsive">
				<table id="cameraList" class="table table-striped table-dark">
					<thead>
						<tr>
							<th> Camera ID </th>
							<th> Location </th>
							<th> Add Date </th>
							<th> Remove Date </th>
							<th> Status </th>
                            <th style="width:90px;"> Action </th>
						</tr>
					</thead>
					${tableData}
				</table>
			</div>
            </div>
		`;
        document.getElementById('content_container').innerHTML = pageContent;
    });
}