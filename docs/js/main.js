// SABER project site — GUI carousel, 2.5D fan (fully automatic)
// 5 slides visible: center front + 2 angled on each side; new slides
// enter from the right every 3 s.
(function () {
  "use strict";

  var carousel = document.getElementById("carousel");
  var caption = document.getElementById("carousel-caption");
  if (!carousel) return;

  var items = Array.prototype.slice.call(carousel.querySelectorAll(".carousel-item"));
  if (!items.length) return;

  var AUTO_MS = 3000;
  var active = 0;

  var SPREAD = 0.68;    // 相邻卡片横向间距（卡片宽度倍数）
  var SKEW = 0;         // 无斜切：只有大小差，无倾斜
  var SCALES = [1.25, 0.82, 0.68];
  var OPACITIES = [1, 0.9, 0.75];
  var BRIGHTNESS = [1, 0.97, 0.93];

  function layout() {
    var n = items.length;
    items.forEach(function (el, i) {
      var raw = i - active;
      var d = ((raw % n) + n) % n;
      if (d > n / 2) d -= n;       // 归一化到 -n/2..n/2，新片从右侧进入
      var abs = Math.abs(d);
      var st = el.style;
      if (abs > 2) {
        st.opacity = "0";
        st.pointerEvents = "none";
        st.zIndex = "0";
        return;
      }
      var w = el.offsetWidth || 300;   // offsetWidth 不含 transform，避免缩放影响间距基准
      var x = d * SPREAD * w;      // 横向展开
      var scale = SCALES[abs];
      st.transform =
        "translate(-50%, -50%) translateX(" + x + "px) " +
        "scale(" + scale + ") skewY(" + (-d * SKEW) + "deg)";
      st.opacity = String(OPACITIES[abs]);
      st.filter = "brightness(" + BRIGHTNESS[abs] + ")";
      st.zIndex = String(10 - abs);
      st.pointerEvents = abs === 0 ? "auto" : "none";
    });
    if (caption) {
      caption.textContent = items[active].getAttribute("data-caption") || "";
    }
  }

  setInterval(function () {
    active = (active + 1) % items.length;
    layout();
  }, AUTO_MS);

  window.addEventListener("resize", layout);
  layout();
})();
