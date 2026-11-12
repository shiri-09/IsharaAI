"""Demo websites for the browser agent to operate (served by this app, no external service involved).

/demo/canteen  a campus canteen: menu, cart, checkout form (name, phone, block and room, notes), order confirmation
/demo/portal   a faculty portal: request a meeting (professor, date, time, purpose)

They are ordinary pages with ordinary labels, buttons and forms, so the agent drives them the same way it would drive any site.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/demo", tags=["demo sites"], include_in_schema=False)

STYLE = """
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial,sans-serif;background:#0b0c0e;color:#e8eaee}
header{display:flex;align-items:center;gap:12px;padding:16px 28px;border-bottom:1px solid #23252b;background:#111216}
header b{font-size:20px;letter-spacing:.01em}header span{color:#8b919b;font-size:13px}
main{max-width:1000px;margin:0 auto;padding:24px 28px}h2{margin:0 0 14px;font-size:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
.card{border:1px solid #2a2d34;border-radius:14px;padding:16px;background:#14161a}.card h3{margin:0 0 4px;font-size:17px}.card p{margin:0 0 12px;color:#9aa0ab;font-size:14px}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px}.price{font-weight:600}
button,.btn{cursor:pointer;border:1px solid #3a3e47;background:#1c1f25;color:#f1f3f6;border-radius:10px;padding:9px 16px;font-size:14px}
button.primary{background:#e9ecf1;color:#0b0c0e;border-color:#e9ecf1;font-weight:600}button:disabled{opacity:.4;cursor:not-allowed}
.cart{margin-top:26px;border:1px solid #2a2d34;border-radius:14px;padding:16px;background:#101216}.cart li{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2127}
form{display:grid;gap:14px;max-width:520px}label{display:grid;gap:6px;font-size:14px;color:#c4c8cf}
input,select,textarea{background:#0e1013;border:1px solid #30343c;border-radius:10px;color:#f1f3f6;padding:11px 12px;font-size:15px}
.ok{border:1px solid #1f6f4a;background:#0f2a1e;border-radius:14px;padding:22px;max-width:560px}.hidden{display:none}small{color:#8b919b}
"""

CANTEEN = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Ishara Demo Canteen</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>""" + STYLE + """</style></head>
<body><header><b>Ishara Demo Canteen</b><span>A demo site for the Ishara AI browser agent. No real food, no real payment.</span></header>
<main>
<section id="menu"><h2>Today's menu</h2><div class="grid" id="items"></div>
<div class="cart"><h2>Your cart</h2><ul id="cart" style="list-style:none;padding:0;margin:0 0 12px"></ul><div class="row"><b>Total: <span id="total">Rs 0</span></b><button class="primary" id="checkout" disabled>Checkout</button></div></div></section>
<section id="form" class="hidden"><h2>Delivery details</h2>
<form id="order-form" onsubmit="return false">
<label>Full name<input id="name" name="name" autocomplete="name" placeholder="Your full name"></label>
<label>Phone number<input id="phone" name="phone" inputmode="tel" placeholder="10 digit mobile number"></label>
<label>Block and room<input id="room" name="room" placeholder="e.g. Block B, Room 214"></label>
<label>Delivery notes (optional)<textarea id="notes" name="notes" rows="2" placeholder="e.g. less spicy, call at the gate"></textarea></label>
<label>Payment<select id="pay" name="payment"><option>Cash on delivery</option></select></label>
<div class="row" style="justify-content:flex-start"><button type="button" id="back">Back to menu</button><button type="button" class="primary" id="place">Place order</button></div>
<small id="err" style="color:#f87171"></small></form></section>
<section id="done" class="hidden"><div class="ok"><h2>Order placed</h2><p id="summary"></p><p><b>Order number: <span id="num"></span></b></p><p><small>This is a demo. Nothing was charged and no food will arrive.</small></p></div></section>
</main>
<script>
const MENU=[["Veg Biryani","Fragrant rice with seasonal vegetables",120],["Paneer Wrap","Grilled paneer, onion and mint chutney",90],["Masala Dosa","Crisp dosa with potato masala and chutneys",70],["Idli Vada","Two idlis and a vada with sambar",50],["Filter Coffee","South Indian filter coffee",30],["Fruit Salad","Fresh seasonal fruit",60]];
const cart={};const $=id=>document.getElementById(id);
function render(){const ul=$("cart");ul.innerHTML="";let t=0;for(const [n,q] of Object.entries(cart)){const p=MENU.find(m=>m[0]===n)[2];t+=p*q;const li=document.createElement("li");li.innerHTML="<span>"+n+" x "+q+"</span><span>Rs "+(p*q)+"</span>";ul.appendChild(li)}
if(!Object.keys(cart).length){ul.innerHTML="<li><small>Your cart is empty</small></li>"}$("total").textContent="Rs "+t;$("checkout").disabled=!t}
MENU.forEach(([n,d,p])=>{const c=document.createElement("div");c.className="card";c.innerHTML="<h3>"+n+"</h3><p>"+d+"</p><div class='row'><span class='price'>Rs "+p+"</span><button aria-label='Add "+n+" to cart'>Add to cart</button></div>";c.querySelector("button").onclick=()=>{cart[n]=(cart[n]||0)+1;render()};$("items").appendChild(c)});
$("checkout").onclick=()=>{$("menu").classList.add("hidden");$("form").classList.remove("hidden");window.scrollTo(0,0)};
$("back").onclick=()=>{$("form").classList.add("hidden");$("menu").classList.remove("hidden")};
$("place").onclick=()=>{const nm=$("name").value.trim(),ph=$("phone").value.trim(),rm=$("room").value.trim();if(!nm||!ph||!rm){$("err").textContent="Please fill in your name, phone number and block and room.";return}
$("err").textContent="";const num="CN-"+(1000+Math.floor(Math.random()*9000));const items=Object.entries(cart).map(([n,q])=>n+" x "+q).join(", ");
$("summary").textContent=items+" for "+nm+", "+rm+", phone "+ph+(($("notes").value.trim())?(". Notes: "+$("notes").value.trim()):"")+".";$("num").textContent=num;$("form").classList.add("hidden");$("done").classList.remove("hidden");window.scrollTo(0,0)};
render();
</script></body></html>"""

PORTAL = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Ishara Demo Faculty Portal</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>""" + STYLE + """</style></head>
<body><header><b>Faculty Portal</b><span>A demo site for the Ishara AI browser agent.</span></header>
<main><section id="form"><h2>Request a meeting</h2>
<form onsubmit="return false">
<label>Your name<input id="name" name="name" placeholder="Your full name"></label>
<label>Your phone number<input id="phone" name="phone" placeholder="10 digit mobile number"></label>
<label>Professor<select id="prof" name="professor"><option value="">Choose a professor</option><option>Professor Rao</option><option>Professor Sharma</option><option>Professor Iyer</option></select></label>
<label>Preferred day<select id="day" name="day"><option value="">Choose a day</option><option>Monday</option><option>Tuesday</option><option>Wednesday</option><option>Thursday</option><option>Friday</option></select></label>
<label>Preferred time<select id="time" name="time"><option value="">Choose a time</option><option>10:00 AM</option><option>11:00 AM</option><option>2:00 PM</option><option>3:00 PM</option></select></label>
<label>What is it about?<textarea id="about" name="about" rows="2" placeholder="Short reason"></textarea></label>
<button class="primary" id="send" type="button">Request meeting</button><small id="err" style="color:#f87171"></small></form></section>
<section id="done" class="hidden"><div class="ok"><h2>Request sent</h2><p id="summary"></p><p><b>Reference: <span id="num"></span></b></p><p><small>This is a demo. No professor was contacted.</small></p></div></section></main>
<script>const $=id=>document.getElementById(id);
$("send").onclick=()=>{const v=k=>$(k).value.trim();if(!v("name")||!v("phone")||!v("prof")||!v("day")||!v("time")){$("err").textContent="Please fill in your name, phone, professor, day and time.";return}
$("summary").textContent="Meeting with "+v("prof")+" on "+v("day")+" at "+v("time")+" for "+v("name")+(v("about")?(". About: "+v("about")):"")+".";$("num").textContent="MT-"+(1000+Math.floor(Math.random()*9000));$("form").classList.add("hidden");$("done").classList.remove("hidden")};</script></body></html>"""


@router.get("/canteen", response_class=HTMLResponse)
def canteen():
    return CANTEEN


@router.get("/portal", response_class=HTMLResponse)
def portal():
    return PORTAL
