// Icon swap is handled entirely by CSS ([data-theme="dark"] selectors in
// styles.css) reacting to the data-theme attribute, so this only needs to
// flip that attribute and persist the choice. Initial theme is applied by
// an inline script in <head> so there's no flash of the wrong theme while
// this deferred file loads.
function initThemeToggle() {
  var btn = document.querySelector('.theme-toggle');
  if (!btn) return;

  btn.addEventListener('click', function () {
    var current = document.documentElement.getAttribute('data-theme') || 'light';
    var next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
  });
}

document.addEventListener('DOMContentLoaded', initThemeToggle);

