const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('.nav-links');
function closeMenu() { nav.classList.remove('open'); menu.setAttribute('aria-expanded', 'false'); }
menu.addEventListener('click', () => { const open = menu.getAttribute('aria-expanded') !== 'true'; menu.setAttribute('aria-expanded', String(open)); nav.classList.toggle('open', open); });
nav.querySelectorAll('a').forEach(a => a.addEventListener('click', closeMenu));
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });
document.addEventListener('click', e => { if (!e.target.closest('.nav')) closeMenu(); });
window.matchMedia('(min-width:681px)').addEventListener('change', closeMenu);

const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab, focus = false) {
  for (const item of tabs) { const active = item === tab; item.setAttribute('aria-selected', String(active)); item.tabIndex = active ? 0 : -1; document.getElementById(item.getAttribute('aria-controls')).hidden = !active; }
  if (focus) tab.focus();
}
tabs.forEach((tab, i) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', e => {
    const next = e.key === 'ArrowRight' ? (i + 1) % tabs.length : e.key === 'ArrowLeft' ? (i + tabs.length - 1) % tabs.length : e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : -1;
    if (next >= 0) { e.preventDefault(); selectTab(tabs[next], true); }
  });
});

const dialog = document.getElementById('figure-dialog');
const dialogImage = document.getElementById('dialog-image');
document.querySelectorAll('[data-figure]').forEach(button => button.addEventListener('click', () => {
  const src = button.dataset.figure;
  if (!dialog.showModal) { window.open(src, '_blank', 'noopener'); return; }
  dialogImage.src = src;
  dialogImage.alt = button.querySelector('img').alt;
  document.getElementById('figure-title').textContent = button.dataset.title;
  document.getElementById('original-figure').href = src;
  dialog.showModal();
  document.body.style.overflow = 'hidden';
  document.querySelector('.dialog-image-wrap').scrollTo(0, 0);
}));
document.getElementById('close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => { document.body.style.overflow = ''; });
dialog.addEventListener('click', e => { if (e.target === dialog) { const r = dialog.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close(); } });

const copyButton = document.getElementById('copyBib');
const bibtex = document.getElementById('bibtex');
const copyStatus = document.getElementById('copy-status');
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent);
    copyButton.textContent = 'Copied!'; copyStatus.textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2200);
  } catch {
    const range = document.createRange(); range.selectNodeContents(bibtex); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    let copied = false;
    try { copied = document.execCommand('copy'); } catch {}
    if (copied) {
      selection.removeAllRanges(); copyButton.textContent = 'Copied!'; copyStatus.textContent = 'BibTeX copied to clipboard.';
      setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2200);
    } else {
      copyStatus.textContent = 'BibTeX selected. Press Ctrl+C or Command+C to copy, or download the .bib file.';
    }
  }
});

if ('IntersectionObserver' in window) {
  const links = [...document.querySelectorAll('.nav-links a')];
  const observer = new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) links.forEach(a => a.classList.toggle('active', a.hash === '#' + entry.target.id)); }), {rootMargin:'-20% 0px -65% 0px'});
  document.querySelectorAll('main section[id]').forEach(s => observer.observe(s));
}
