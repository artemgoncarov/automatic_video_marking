document.getElementById('videoInput').addEventListener('change', function () {
    document.getElementById('videobtn').style.display = 'none';
    const videoInput = document.getElementById('videoInput');
    const videoFile = videoInput.files[0];

    if (videoFile) {
        const formData = new FormData();
        formData.append('video', videoFile);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }

                // Скрываем инпут и показываем загруженное видео
                document.getElementById('videoInput').style.display = 'none';
                document.getElementById('uploadedVideo').style.display = 'block';
                document.getElementById('videoSource').src = data.video_url;
                document.getElementById('uploadedVideo').load(); // Перезагружаем видео

                // Сохраняем URL загруженного видео
                window.videoUrl = data.video_url;
                document.getElementById('videoContainer').style.display = 'block';
            })
            .catch(error => console.error('Error:', error));
    }
});

document.getElementById('getInfoButton').addEventListener('click', () => {
    const loadingContainer = document.getElementById('loadingContainer');
    loadingContainer.style.display = 'block';
    document.querySelector('#getInfoContainer').style.display = 'none';
    fetch(`/getInfo?video_path=${window.videoUrl}&second=${document.getElementById('uploadedVideo').currentTime}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            loadingContainer.style.display = 'none';
            document.querySelector('#getInfoContainer').style.display = 'block';
            document.querySelector('#getInfoText1').textContent = `Описание: ${data.description[0]}`;
            document.querySelector('#getInfoText2').textContent = `Сцена: ${data.description[1]}`;
            document.querySelector('#getInfoText3').textContent = `Риски: ${data.description[2]}`;
            document.querySelector('#getInfoText4').textContent = `Распознанный текст: ${data.text}`;
            // document.querySelector("#tempImg").src = data.path;
            console.log(data.description);
            const downloadContainer = document.getElementById('getInfoContainer');
            const downloadBtn = document.getElementById('getInfoText5');
            downloadBtn.className = 'btn btn-primary mt-4';
            downloadBtn.href = `/tempImg`; // Устанавливаем ссылку для скачивания файла
            downloadBtn.innerText = 'Скачать теплограмму';
            // downloadContainer.appendChild(downloadBtn);
        })
})

document.getElementById('getInfoByContext').addEventListener('click', () => {
    document.getElementById('getInfoByContextContainer').style.display = 'block';
})


document.getElementById('getInfoByContextBtn').addEventListener('click', () => {
    const text = document.getElementById('contextInput').textContent;
    const loadingContainer = document.getElementById('loadingContainer');
    loadingContainer.style.display = 'block';

    fetch(`/getSecond?text=${text}&video=${window.videoUrl}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json()).then(data => {
        loadingContainer.style.display = 'none';
        document.getElementById('resultDiv').style.display = 'block';
        document.getElementById('resultDiv').textContent = `Возможные time-stamp: ${data.second1}, ${data.second2}, ${data.second3} сек.`;
    })
})

document.getElementById('submitButton').addEventListener('click', function () {
    const videoUrl = window.videoUrl;

    if (videoUrl) {
        // Скрываем загруженное видео
        // document.getElementById('uploadedVideo').style.display = 'none';
        document.getElementById('submitButton').style.display = 'none';

        // Начинаем отображать потоковое видео
        const videoStream = document.getElementById('videoStream');
        videoStream.src = `/video_feed?video_url=${encodeURIComponent(videoUrl)}`;

        // Показываем контейнер с кадрами
        document.getElementById('framesContainer').style.display = 'block';

        // Показываем анимацию загрузки и текст
        const loadingContainer = document.getElementById('loadingContainer');
        loadingContainer.style.display = 'block';

        // Запрос для обнаружения объектов
        fetch('/detect_objects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ video_url: videoUrl })
        })
            .then(response => response.json())
            .then(data => {
                const durationContainer = document.getElementById('durationContainer');
                durationContainer.innerHTML = ''; // Очищаем предыдущие результаты

                // Создаем карточки для каждого обнаруженного объекта
                for (const [key, value] of Object.entries(data.detected_objects.objects)) {
                    const card = document.createElement('div');
                    card.className = 'card m-2';
                    card.style.width = '18rem';

                    const imgSrc = `static/images/${key}.png`; // Путь к изображению класса
                    card.innerHTML = `
                    <div class="card-body">
                        <h5 class="card-title">${key}</h5>
                        <p class="card-text">Количество кадров: ${value.frame_count}</p>
                        <p class="card-text">Процент кадров: ${Math.round(value.frame_count / data.detected_objects.total_frames * 100)}%</p>
                        <p class="card-text">Обнаружено: ${value.total_count} раз(а)</p>
                    </div>
                `;
                    // <img src="${imgSrc}" class="card-img-top" alt="${key}"></img>

                    durationContainer.appendChild(card);
                }

                // Запрос для генерации транскрипции
                fetch('/generate_transcription', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ video_url: videoUrl })
                })
                    .then(() => {
                        // Скрываем анимацию и текст, когда кнопка появляется
                        loadingContainer.style.display = 'none';

                        // Показываем контейнер для скачивания
                        const downloadContainer = document.getElementById('downloadContainer');
                        downloadContainer.style.display = 'block';

                        // Очищаем контейнер перед добавлением новой кнопки
                        downloadContainer.innerHTML = '';

                        // Добавляем кнопку для скачивания CSV файла
                        const downloadBtn = document.createElement('a');
                        downloadBtn.className = 'btn btn-primary mt-4';
                        downloadBtn.href = `/transcribe`; // Устанавливаем ссылку для скачивания файла
                        downloadBtn.innerText = 'Скачать транскрибацию';

                        // Вставляем кнопку в контейнер
                        downloadContainer.appendChild(downloadBtn);

                        const downloadAnalysBtn = document.createElement('a');
                        downloadAnalysBtn.className = 'btn btn-primary mt-4';
                        downloadAnalysBtn.href = `/report`; // Устанавливаем ссылку для скачивания файла
                        downloadAnalysBtn.innerText = 'Скачать анализ';

                        // Вставляем кнопку в контейнер
                        downloadContainer.appendChild(downloadAnalysBtn);

                        const downloadAudioBtn = document.createElement('a');
                        downloadAudioBtn.className = 'btn btn-primary mt-4';
                        downloadAudioBtn.href = `/audio`; // Устанавливаем ссылку для скачивания файла
                        downloadAudioBtn.innerText = 'Скачать анализ аудио';

                        // Вставляем кнопку в контейнер
                        downloadContainer.appendChild(downloadAudioBtn);

                    })
                    .catch(error => {
                        console.error('Error generating transcription:', error);
                        loadingContainer.style.display = 'none'; // Скрыть загрузку при ошибке
                    });

                // После генерации транскрипции выполняем анализ текста
            })
            .catch(error => console.error('Error detecting objects:', error));
    }
});
