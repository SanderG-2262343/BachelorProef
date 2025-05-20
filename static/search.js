document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('project-search');
    const suggestionsBox = document.getElementById('suggestions');
    const projIdInput = document.getElementById('proj-id');
    const form = document.getElementById('project-form');
    const resultsDiv = document.getElementById('results');

    searchInput.addEventListener('input', async () => {
        const query = searchInput.value.trim();
        if (query.length < 1) {
            suggestionsBox.innerHTML = '';
            suggestionsBox.setAttribute('hidden', '');
            return;
        }
        suggestionsBox.removeAttribute('hidden');

        const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();

        suggestionsBox.innerHTML = '';
        data.forEach(project => {
            const item = document.createElement('div');
            item.textContent = project.title;
            item.dataset.id = project.projId;
            item.addEventListener('click', () => {
                searchInput.value = project.title;
                projIdInput.value = project.projId;
                suggestionsBox.innerHTML = '';
                suggestionsBox.setAttribute('hidden', '');
            });
            suggestionsBox.appendChild(item);
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const projId = projIdInput.value;
        const k = document.getElementById('k').value;

        if (!projId) {
            alert("Please select a project from the list.");
            return;
        }

        resultsDiv.innerHTML = 'Loading...';

        const res = await fetch('/correlated', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ projId, k })
        });

        const html = await res.text();
        resultsDiv.innerHTML = html;
    });

    // Dismiss suggestions on outside click
    document.addEventListener('click', e => {
        if (!suggestionsBox.contains(e.target) && e.target !== searchInput) {
            suggestionsBox.innerHTML = '';
            suggestionsBox.setAttribute('hidden', '');
        }
    });
});