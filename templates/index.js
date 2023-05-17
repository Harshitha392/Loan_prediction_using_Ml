var div = document.getElementById( 'prediction_text' );
console.log(div)
var div2=document.getElementById('prediction')
if (div=="Loan is approved") {
    div2.style.color = 'green';
}
else{
    div2.style.color="red";
}