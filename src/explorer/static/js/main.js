(() => {
    let abortController = null;

    function labelLines(lines, targetElement) {
        let table = document.createElement('table');
        table.classList.add('labeled-lines');
        for (let l in lines) {
            let tr = document.createElement('tr');
            let tdCls = document.createElement('td');
            let tdBody = document.createElement('td');

            if (lines[l][0] instanceof Array) {
                labelLines(lines[l], tdBody);
            } else {
                let className = lines[l][1].replace(/_/g, ' ');
                let classNameClass = lines[l][1].replace(/</g, '').replace(/>/g, '').replace(/_/g, '-');

                tr.classList.add(`line-label-${classNameClass}`);
                tdCls.appendChild(document.createTextNode(className));
                tdBody.appendChild(document.createTextNode(lines[l][0]));
            }
            tr.appendChild(tdCls);
            tr.appendChild(tdBody);
            table.appendChild(tr);
        }
        targetElement.innerHTML = '';
        targetElement.appendChild(table);
    }

    function reformatMail(text, targetElement, callback = null) {
        if (!text) {
            return;
        }

        fetch(API_REFORMAT_URL, {
            method: 'post',
            headers: {'Content-Type': 'text/plain'},
            body: text,
            signal: abortController.signal,
            importance: 'high'
        }).then(response => {
            return response.json();
        }).then(json => {
            labelLines(json, targetElement);
            if (callback !== null) {
                callback();
            }
        });
    }

    function showThread(message_id, callback = null) {
        fetch(API_GET_THREAD_URL + '?message_id=' + encodeURIComponent(message_id), {
            method: 'get',
            signal: abortController.signal,
            importance: 'high'
        }).then(response => {
            return response.json();
        }).then(json => {
            let modal = document.getElementById('modal');
            let modalHeading = modal.getElementsByTagName('h2')[0];
            let modalContent = modal.getElementsByClassName('uk-modal-body')[0];
            modalHeading.innerText = 'Thread View';
            modalContent.innerHTML = '';

            UIkit.modal(modal).show();
            processResults(json, modalContent);

            if (callback !== null) {
                callback();
            }
        });
    }

    function predictLines(messageId, text, targetElement, targetElementForControls,
                          addShowThreadButtons = false, callback = null) {
        fetch(API_PREDICT_LINES_URL, {
            method: 'post',
            headers: {'Content-Type': 'text/plain'},
            body: text,
            signal: abortController.signal,
            importance: 'low'
        }).then(response => {
            return response.json();
        }).then(json => {
            labelLines(json, targetElement);

            let reformatButton = document.createElement('button');
            reformatButton.innerText = 'Reformat';
            reformatButton.classList.add('uk-button', 'uk-button-default', 'uk-margin-small-right');
            reformatButton.addEventListener('click', e => {
                e.preventDefault();

                e.target.innerText = 'Loading...';
                e.target.disabled = true;

                reformatMail(text, targetElement, () => {
                    e.target.parentElement.removeChild(e.target);
                });
            });
            targetElementForControls.appendChild(reformatButton);

            if (addShowThreadButtons) {
                let threadButton = document.createElement('button');
                const buttonText = 'Show Thread';
                threadButton.innerText = buttonText;
                threadButton.classList.add('uk-button', 'uk-button-default');
                threadButton.addEventListener('click', e => {
                    e.preventDefault();

                    e.target.innerText = 'Loading...';
                    e.target.disabled = true;

                    showThread(messageId, () => {
                        e.target.innerText = buttonText;
                        e.target.disabled = false;
                    })
                });
                targetElementForControls.appendChild(threadButton);
            }

            if (callback !== null) {
                callback();
            }
        });
    }

    function headerDict2DefList(dict) {
        let dl = document.createElement('dl');
        dl.classList.add('uk-margin-small-top', 'uk-margin-small-bottom');

        for (let h in dict) {
            if (!dict[h]) {
                continue;
            }
            let dt = document.createElement('dt');
            dt.appendChild(document.createTextNode(h.replace(/_/g, '-') + ':'));
            let dd = document.createElement('dd');
            dd.appendChild(document.createTextNode(dict[h]));
            dl.appendChild(dt);
            dl.appendChild(dd);
        }

        return dl;
    }

    function processResults(hits, targetElement, addShowThreadButtons = false) {
        for (let hit in hits) {
            let source = hits[hit]['_source'];

            let grid = document.createElement('div');
            grid.classList.add('uk-grid', 'uk-grid-medium', 'uk-child-width-1-2');
            grid.dataset.ukGrid = null;

            let warcHeaders = document.createElement('div');
            warcHeaders.classList.add('header-list');
            warcHeaders.appendChild(headerDict2DefList({
                warc_id: hits[hit]['_id'],
                warc_file: source['warc_file'],
                warc_offset: source['warc_offset'],
                timestamp: source['@timestamp'],
                groupname: source['groupname'],
                news_url: source['news_url']
            }));
            grid.appendChild(warcHeaders);

            let mailHeaders = document.createElement('div');
            mailHeaders.classList.add('header-list');
            mailHeaders.appendChild(headerDict2DefList(source['headers']));
            grid.appendChild(mailHeaders);

            let plainTextContainer = document.createElement('div');
            let plainTextControls = document.createElement('div');
            let plainTextBody = document.createElement('pre');
            plainTextBody.classList.add('plaintext-body');

            let intersectionObserver = null;
            if (source['text_plain'].trim()) {
                plainTextBody.innerText = source['text_plain'];
                plainTextBody.dataset.messageId = source['headers']['message_id'];
                plainTextBody.dataset.originalText = source['text_plain'];
                intersectionObserver = new IntersectionObserver((entries) => {
                    // predict lines only when mail contents are within the viewport
                    if (entries[0].isIntersecting) {
                        predictLines(source['headers']['message_id'], source['text_plain'],
                            plainTextBody, plainTextControls, addShowThreadButtons);
                        intersectionObserver.disconnect();
                    }
                }, {
                    threshold: [0],
                    trackVisibility: true,
                    delay: 100
                });
            } else {
                plainTextBody.appendChild(document.createTextNode('<no plaintext content>'));
                plainTextBody.classList.add('no-content');
            }

            plainTextContainer.appendChild(plainTextControls);
            plainTextContainer.appendChild(plainTextBody);
            grid.appendChild(plainTextContainer);

            let html = document.createElement('div');
            html.classList.add('html-body');

            if (source['text_html'].trim()) {
                let shadow = html.attachShadow({mode: 'open'});
                shadow.innerHTML = DOMPurify.sanitize(source['text_html']);
            } else {
                html.appendChild(document.createTextNode('<no html content>'));
                html.classList.add('no-content');
            }
            grid.appendChild(html);

            let container = document.createElement('div');
            container.classList.add('query-result', 'uk-margin-large-bottom');
            container.appendChild(grid);
            targetElement.appendChild(container);
            if (null !== intersectionObserver) {
                intersectionObserver.observe(plainTextContainer);
            }
        }
    }

    addEventListener('DOMContentLoaded', () => {
        const queryButton = document.getElementById('query-button');
        const querySpinner = document.getElementById('query-spinner');
        const queryResults = document.getElementById('query-results');
        const queryResultsTotal = document.getElementById('query-results-total');

        const editor = ace.edit('query-editor');
        let JsonMode = ace.require("ace/mode/json").Mode;
        editor.session.setMode(new JsonMode());
        editor.session.setUseWrapMode(true);
        editor.commands.addCommand({
            name: 'submit',
            bindKey: {win: 'Ctrl-Enter', mac: 'Command-Return'},
            exec: () => {
                queryButton.click();
            }
        });

        queryButton.addEventListener('click', e => {
            querySpinner.classList.remove('uk-hidden');

            if (abortController !== null) {
                abortController.abort();
            }
            abortController = new AbortController();

            fetch(API_QUERY_MAILS_URL, {
                method: 'post',
                headers: {'Content-Type': 'application/json'},
                body: editor.getValue(),
                signal: abortController.signal
            }).then(response => {
                return response.json();
            }).then(json => {
                queryResults.innerHTML = '';
                queryResultsTotal.innerHTML = `<strong>Total results:</strong> ${json['total']}`;
                processResults(json['hits'], queryResults, true);
                querySpinner.classList.add('uk-hidden');
            }).catch(e => {
                console.info("Fetch aborted: " + e.message);
            });
        });
    });
})();
