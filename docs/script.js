// Lightweight enhancements for the static CLV project site.
// 1) Highlight the active nav link as you scroll.
// 2) Smooth-scroll is handled by CSS (scroll-behavior:smooth).

(function () {
  const links = Array.from(document.querySelectorAll('.nav a.link'));
  const map = new Map();
  links.forEach(a => {
    const id = a.getAttribute('href').slice(1);
    const sec = document.getElementById(id);
    if (sec) map.set(sec, a);
  });

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => (l.style.color = ''));
        const active = map.get(e.target);
        if (active) active.style.color = '#fff';
      }
    });
  }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });

  map.forEach((_, sec) => obs.observe(sec));

  console.log('CLV project site loaded ·', map.size, 'sections');
})();
