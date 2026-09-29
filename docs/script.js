const copyButton = document.getElementById('copyBib');
const bibtex = document.getElementById('bibtex');

if (copyButton && bibtex) {
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent);
      copyButton.textContent = 'Copied!';
      setTimeout(() => { copyButton.textContent = 'Copy'; }, 1500);
    } catch (_) {
      copyButton.textContent = 'Select text';
    }
  });
}
