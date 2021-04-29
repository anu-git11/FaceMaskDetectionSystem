function onSearch() {
	let input, filter, table, tr, td, cell, i, j;
	input = document.getElementById("detailFilter");
	filter = input.value.toUpperCase();
	table = document.getElementById("reportTable");
	tr = table.getElementsByTagName("tr");

	// Loop through all table rows, and hide those who don't match the search query
	for (i = 1; i < tr.length; i++) {
		// Hide the row initially.
		tr[i].style.display = "none";

		td = tr[i].getElementsByTagName("td");
		for (j = 0; j < td.length; j++) {
			cell = tr[i].getElementsByTagName("td")[j];
			if (cell) {
				if (cell.innerHTML.toUpperCase().indexOf(filter) > -1) {
					tr[i].style.display = "";
					break;
				}
			}
		}
	}
}

function loadReport(selectedYear = undefined, selectedWeek = undefined) {

	let pageTitle = selectedWeek ? `Week ${selectedWeek} Reports` : "Weekly Reports";
	let tableContainer;
	let parsedYearWeek = `${selectedYear}-W${selectedWeek}`;

	let pageContent = `
			<input type="week" id="dateFilter" oninput="onDateChange()">
			<h1 class="text-center"> ${pageTitle} </h1>
            <div class="table-container" id="table-container" style="display: none;">
            	<div class ="row">
            		<div class = "col-md-2"></div>
					<div class = "col-md-4" id="chart1_div"></div>
					<div class = "col-md-4" id="chart2_div"></div>
					<div class = "col-md-2"></div>
				</div>
				<div class ="row">
					
					<input type="text" id="detailFilter" class="table-search" placeholder="Search here" onkeyup="onSearch()">
					
				</div>
				<div class="table-responsive" id="table-responsive"></div>
            </div>
		`;

	document.getElementById('content_container').innerHTML = pageContent;

	if (selectedWeek && selectedYear) {
		
		$.ajax({
			url: "repository/getReport.php",
			data: { 
				"selectedWeek": selectedWeek,
				"selectedYear": selectedYear
			}
		}).then(function (data) {
			let tableData = "";
			
			if (data) {
				
				data = JSON.parse(data);
				if (data['overallStats']) {
					loadCharts(data['overallStats']);
				}
				if (data['employeeDetails']) {
					let columns = Object.keys(data['employeeDetails'][0]);
					for (var rowName = 0; rowName < data['employeeDetails'].length; rowName++) {
						let rowData = '<tr>';
						columns.forEach(
							(colName) => {
								let cellData = String(data['employeeDetails'][rowName][colName]);
								// replace underscore to space
								rowData += `
								<td> 
									<div id="${colName}"> ${cellData.replace(/_/g, ' ')} </div>
								</td>`
							}
						)
						rowData += '</tr>';
						tableData += rowData;
					}
				}
			}
	
			if(tableData === "") {
				tableData += '<tr> <td colspan="9"> No Data </td> </tr>';
			}
	
			let tableContent = `
				
						<table class="sortable table table-striped table-dark" id="reportTable">
							<thead>
								<tr>
									<th class="sorttable_sorted"> ID <span id="sorttable_sortfwdind">&nbsp;▾</span></th>
									<th> Name </th>
									<th> Title </th>
									<th> Email </th>
									<th> Phone </th>
									<th> Capture Date </th>
									<th> Capture Time </th>
									<th> Camera Location </th>
								</tr>
							</thead>
							${tableData}
						</table>
			`;
	
			document.getElementById('table-responsive').innerHTML = tableContent;
			return parsedYearWeek;
	
		})
		.then((parsedYearWeek) => {
			document.getElementById('dateFilter').value = parsedYearWeek;
	
			// show/hide chart and table base on week selection
			tableContainer = document.getElementById("table-container");
			if (selectedWeek && selectedYear) {
				tableContainer.style.display = "block";
			} else {
				console.log('hidding')
				tableContainer.style.display = "none";
			}
	
			// make table sortable by clicking on header
			let reportTable = document.getElementById("reportTable");
			sorttable.makeSortable(reportTable);
		});
	}
}

function loadCharts(data) {
	// Load the Visualization API and the corechart package.
	google.charts.load('current', { 'packages': ['corechart'] });

	// Set a callback to run when the Google Visualization API is loaded.
	google.charts.setOnLoadCallback(drawEmployeeChart);

	// Callback that creates and populates a data table,
	// instantiates the pie chart, passes in the data and
	// draws it.
	function drawEmployeeChart() {

		// ----- first chart ---------
		// Create the data table.
		var dataTable = google.visualization.arrayToDataTable([
			['Type', 'Count'],
			['Masked', data['total_employee'] - data['total_result_employee']],
			['Unmasked', data['total_result_employee']]
		]);

		// Set chart options
		var options = {
			'width': 500,
			'height': 400,
			backgroundColor: 'transparent',colors: ['#809cb5', '#34485c'],
			legend:{textStyle:{fontSize:'14', fontName: 'Montserrat'}}
		};
		var title = { 'title': 'Employee Ratio' , 
					  titleTextStyle: {fontSize:'16', fontName: 'Montserrat'}

				};

		// Instantiate and draw our chart, passing in some options.
		var chart = new google.visualization.PieChart(document.getElementById('chart1_div'));
		chart.draw(dataTable, { ...title, ...options });


		// ----- second chart ---------
		dataTable = google.visualization.arrayToDataTable([
			['Type', 'Count'],
			['Visitor', data['total_result'] - data['total_result_employee_occurance']],
			['Employee', data['total_result_employee_occurance']]
		]);
		title = { 'title': 'Unmasked Occurance Ratio' ,
				titleTextStyle: {fontSize:'16', fontName: 'Montserrat'}};
		chart = new google.visualization.PieChart(document.getElementById('chart2_div'));
		chart.draw(dataTable, { ...title, ...options });
	}
}

function onDateChange() {
	var selectedWeek = document.getElementById("dateFilter").value;
	
	let [year, week] = selectedWeek.split('-');
	week = week.slice(1);
	this.loadReport(year, week);
}