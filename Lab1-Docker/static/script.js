const contentDiv = document.getElementById('book-content');

// Questionnaire State
let state = {
    step: 0,
    answers: {}
};

// Questions Configuration
const questions = [
    {
        id: 'name',
        text: "Who is this thought for?",
        type: 'text',
        placeholder: "Their Name"
    },
    {
        id: 'relation',
        text: "How do you know them?",
        type: 'select',
        options: ["Friend", "Partner", "Family", "Colleague"]
    },
    {
        id: 'vibe',
        text: "What's the vibe?",
        type: 'select',
        options: ["Sentimental", "Funny", "Motivational", "Simple"]
    }
];

// Message Templates based on 'vibe' (Arrays for variety)
const templates = {
    Sentimental: [
        (name) => `To ${name}, just wanted to remind you how much you mean to me.`,
        (name) => `${name}, the world is a better place with you in it.`,
        (name) => `Forever grateful to have you in my life, ${name}.`
    ],
    Funny: [
        (name) => `Hey ${name}, hope you're doing better than my plants (they're dead).`,
        (name) => `${name}, you're my favorite human (don't tell the dog).`,
        (name) => `I'd share my fries with you, ${name}. That's big.`
    ],
    Motivational: [
        (name) => `${name}, keep crushing it. You're doing amazing!`,
        (name) => `Believe in yourself, ${name}. You've got this.`,
        (name) => `Sending you some power, ${name}. Go get 'em!`
    ],
    Simple: [
        (name) => `Thinking of you, ${name}.`,
        (name) => `Just saying hi, ${name}!`,
        (name) => `Sending a smile your way, ${name}.`
    ]
};

function renderStep() {
    // Clear current content
    contentDiv.innerHTML = '';

    // Animation fade helper
    contentDiv.style.opacity = 0;

    setTimeout(() => {
        if (state.step < questions.length) {
            renderQuestion(questions[state.step]);
        } else {
            renderFinalMessage();
        }
        contentDiv.style.opacity = 1;
    }, 200);
}

function renderQuestion(q) {
    const label = document.createElement('p');
    label.className = 'question-text';
    label.textContent = q.text;

    let input;
    if (q.type === 'text') {
        input = document.createElement('input');
        input.type = 'text';
        input.placeholder = q.placeholder;
        input.className = 'form-input';
    } else if (q.type === 'select') {
        input = document.createElement('div');
        input.className = 'options-grid';

        q.options.forEach(opt => {
            const btn = document.createElement('button');
            btn.className = 'option-btn';
            btn.textContent = opt;
            btn.onclick = () => handleAnswer(q.id, opt);
            input.appendChild(btn);
        });
    }

    contentDiv.appendChild(label);

    if (q.type === 'text') {
        contentDiv.appendChild(input);

        const nextBtn = document.createElement('button');
        nextBtn.textContent = "Next";
        nextBtn.onclick = () => {
            if (input.value.trim()) handleAnswer(q.id, input.value);
        };
        contentDiv.appendChild(nextBtn);
    } else {
        contentDiv.appendChild(input);
    }
}

function handleAnswer(id, value) {
    state.answers[id] = value;
    state.step++;
    renderStep();
}

function renderFinalMessage() {
    const { name, vibe } = state.answers;

    // Select random template for the vibe
    const options = templates[vibe] || templates.Simple;
    const template = options[Math.floor(Math.random() * options.length)];
    const message = template(name);

    const msgEl = document.createElement('p');
    msgEl.id = 'dynamic-message';
    msgEl.textContent = message;

    const restartBtn = document.createElement('button');
    restartBtn.textContent = "Send Another";
    restartBtn.style.marginTop = "20px";
    restartBtn.onclick = () => {
        state.step = 0;
        state.answers = {};
        renderStep();
    };

    contentDiv.appendChild(msgEl);
    contentDiv.appendChild(restartBtn);
}

// Initialize
renderStep();
