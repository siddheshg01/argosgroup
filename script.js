const askForm = document.querySelector('#ask-form');
const askInput = document.querySelector('#ask-input');
const askResponse = document.querySelector('#ask-response');

const answers = [
  'I found it: one $8,420 payment to Northstar Labs was 3× its usual amount. I matched it to the software procurement policy for review.',
  'Based on this month’s revenue and burn, net cash flow is projected to reach $72k next month. Your runway remains healthy at about 8 months.',
  'Revenue has grown for three straight months, led by Shopify payouts. Expenses are up modestly; one vendor payment explains most of the change.'
];

askForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = askInput.value.trim();
  if (!question) {
    askInput.focus();
    return;
  }
  const q = question.toLowerCase();
  askResponse.textContent = q.includes('forecast') || q.includes('next') ? answers[1] : q.includes('expense') || q.includes('burn') ? answers[2] : answers[0];
  askInput.value = '';
});

document.querySelectorAll('.suggestion-chips button').forEach((button) => {
  button.addEventListener('click', () => {
    askInput.value = button.textContent;
    askForm.requestSubmit();
  });
});

document.querySelector('#investigation-button').addEventListener('click', () => {
  const flaggedRow = document.querySelector('.flagged');
  flaggedRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
  flaggedRow.style.backgroundColor = '#fff2ed';
  setTimeout(() => { flaggedRow.style.backgroundColor = ''; }, 1800);
});

document.querySelector('#view-all-button').addEventListener('click', (event) => {
  const hidden = document.querySelector('.hidden-row');
  const isOpen = hidden.classList.toggle('show');
  event.currentTarget.innerHTML = isOpen ? 'Show less <span>↑</span>' : 'View all <span>→</span>';
});

document.querySelector('#date-button').addEventListener('click', (event) => {
  const label = event.currentTarget.querySelector('span');
  label.textContent = label.textContent === 'This month' ? 'Last month' : 'This month';
});

const themeToggle = document.querySelector('#theme-toggle');
const savedTheme = localStorage.getItem('finop-theme') || 'light';
document.documentElement.dataset.theme = savedTheme;
function updateThemeToggle() {
  const isDark = document.documentElement.dataset.theme === 'dark';
  themeToggle.setAttribute('aria-label', `Switch to ${isDark ? 'light' : 'dark'} mode`);
  themeToggle.title = `Switch to ${isDark ? 'light' : 'dark'} mode`;
  themeToggle.querySelector('.theme-icon').textContent = isDark ? '☀' : '☾';
  themeToggle.querySelector('.theme-label').textContent = isDark ? 'Light' : 'Dark';
}
updateThemeToggle();
themeToggle.addEventListener('click', () => {
  const nextTheme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = nextTheme;
  localStorage.setItem('finop-theme', nextTheme);
  updateThemeToggle();
});
