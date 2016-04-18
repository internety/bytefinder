function highlight() {
	var form=document.forms[0];
	for (i=0;i<form.length;i++) {
		if (form[i].checked) {
			var cat=i;
		}
	}
	var x=document.getElementById("text");
	var text=x.textContent||x.innerText;
	var newText='';
	for(var a=0; a<text.length; a++) {
		var letter = text.charAt(a);
		if (a > 0 && a<=colors.length) {
			var opac = Math.round(Math.min(Math.max(colors[a-1][cat],0.0),1.0));
		}
		else {
			var opac = 0.0;
		}
		newText += '<span style="background-color:rgba(255,255,0,' + opac + ')">' + letter + '</span>';
	}
	x.innerHTML = newText;
}