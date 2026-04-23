let board = ["", "", "", "", "", "", "", "", ""];
let currentPlayer = "X";
let isGameActive = true;
let gameMode = "1p"; // 1p vs AI, 2p local
let scores = { human: 0, ai: 0, draw: 0 };
let moveCount = 1;

// DOM Elements
const cells = document.querySelectorAll('.cell');
const statusText = document.getElementById('status-text');
const overlay = document.getElementById('game-overlay');
const overlayTitle = document.getElementById('overlay-title');
const overlaySubtitle = document.getElementById('overlay-subtitle');
const overlayIcon = document.getElementById('overlay-icon');

// Initialization
cells.forEach(cell => cell.addEventListener('click', handleCellClick));

function handleCellClick(e) {
    const cell = e.target;
    const index = cell.getAttribute('data-index');

    if (board[index] !== "" || !isGameActive) return;

    // Human (X) move
    makeMove(index, currentPlayer);

    if (gameMode === "1p" && isGameActive && currentPlayer === "O") {
        statusText.textContent = "AI is thinking...";
        isGameActive = false; // block clicks
        fetchAIMove();
    }
}

function makeMove(index, player) {
    board[index] = player;
    const cell = document.getElementById(`cell-${index}`);
    cell.textContent = player;
    cell.classList.add(player.toLowerCase());
    
    checkGameEnd();
    
    if (isGameActive) {
        currentPlayer = currentPlayer === "X" ? "O" : "X";
        statusText.textContent = gameMode === "1p" 
            ? (currentPlayer === "X" ? "Your turn — place X" : "AI is thinking...") 
            : `Player ${currentPlayer}'s turn`;
    }
}

function fetchAIMove() {
    fetch('/api/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: board })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
            return;
        }

        updateAnalytics(data.metrics);
        isGameActive = true;
        makeMove(data.move, "O");
    })
    .catch(error => {
        console.error("Error fetching AI move:", error);
        isGameActive = true;
        statusText.textContent = "Error: Could not connect to AI.";
    });
}

function checkGameEnd() {
    const winningCombinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ];

    let roundWon = false;
    let winningCells = [];
    let winner = null;

    for (let i = 0; i < winningCombinations.length; i++) {
        const [a, b, c] = winningCombinations[i];
        if (board[a] && board[a] === board[b] && board[a] === board[c]) {
            roundWon = true;
            winningCells = [a, b, c];
            winner = board[a];
            break;
        }
    }

    if (roundWon) {
        endGame(winner);
        winningCells.forEach(idx => document.getElementById(`cell-${idx}`).classList.add('win-cell'));
        return;
    }

    if (!board.includes("")) {
        endGame("Draw");
    }
}

function endGame(winner) {
    isGameActive = false;
    
    if (winner === "Draw") {
        scores.draw++;
        document.getElementById('score-draw').textContent = scores.draw;
        showOverlay("It's a Draw!", "Nobody wins this time.", "🤝");
        statusText.textContent = "Game ended in a draw.";
    } else {
        if (winner === "X") {
            scores.human++;
            document.getElementById('score-human').textContent = scores.human;
            showOverlay("You Win!", "Impressive, you beat the AI!", "🎉");
            statusText.textContent = "You win!";
        } else {
            scores.ai++;
            document.getElementById('score-ai').textContent = scores.ai;
            showOverlay("AI Wins!", "The Minimax algorithm is ruthless.", "🤖");
            statusText.textContent = "AI wins!";
        }
    }
}

function showOverlay(title, subtitle, icon) {
    overlayTitle.textContent = title;
    overlaySubtitle.textContent = subtitle;
    overlayIcon.textContent = icon;
    setTimeout(() => {
        overlay.style.display = 'flex';
    }, 500);
}

function resetGame() {
    board = ["", "", "", "", "", "", "", "", ""];
    currentPlayer = "X";
    isGameActive = true;
    
    cells.forEach(cell => {
        cell.textContent = "";
        cell.className = "cell";
    });
    
    overlay.style.display = 'none';
    statusText.textContent = gameMode === "1p" ? "Your turn — place X" : "Player X's turn";
}

function setGameMode(mode) {
    gameMode = mode;
    document.getElementById('btn-1p').classList.toggle('active', mode === '1p');
    document.getElementById('btn-2p').classList.toggle('active', mode === '2p');
    resetGame();
}

function updateAnalytics(metrics) {
    // Update metrics UI
    document.getElementById('mm-time').textContent = metrics.minimax.time_ms + " ms";
    document.getElementById('mm-nodes').textContent = metrics.minimax.nodes_explored.toLocaleString();
    
    document.getElementById('ab-time').textContent = metrics.alpha_beta.time_ms + " ms";
    document.getElementById('ab-nodes').textContent = metrics.alpha_beta.nodes_explored.toLocaleString();
    
    document.getElementById('comp-speedup').textContent = metrics.speedup + "x";
    document.getElementById('comp-nodes-saved').textContent = metrics.nodes_saved.toLocaleString();
    document.getElementById('comp-prune-pct').textContent = metrics.prune_percentage + "%";
    
    // Update progress bars
    const maxNodes = Math.max(metrics.minimax.nodes_explored, 1);
    const abWidth = (metrics.alpha_beta.nodes_explored / maxNodes) * 100;
    
    document.getElementById('bar-section').style.display = 'block';
    document.getElementById('bar-mm').style.width = '100%';
    document.getElementById('bar-mm-val').textContent = metrics.minimax.nodes_explored;
    document.getElementById('bar-ab').style.width = abWidth + '%';
    document.getElementById('bar-ab-val').textContent = metrics.alpha_beta.nodes_explored;

    // Add to history
    const historyList = document.getElementById('history-list');
    if (moveCount === 1) {
        historyList.innerHTML = `
            <div class="history-header">
                <span>#</span>
                <span>Minimax</span>
                <span>Alpha-Beta</span>
                <span>Saved</span>
            </div>
        `;
    }
    
    const entry = document.createElement('div');
    entry.className = 'history-item-row';
    entry.innerHTML = `
        <span>${moveCount}</span>
        <span style="color: var(--mm-color);">${metrics.minimax.nodes_explored}</span>
        <span style="color: var(--ab-color);">${metrics.alpha_beta.nodes_explored}</span>
        <span style="color: var(--win-color);">- ${metrics.nodes_saved}</span>
    `;
    historyList.appendChild(entry);
    moveCount++;
}

function clearStats() {
    scores = { human: 0, ai: 0, draw: 0 };
    document.getElementById('score-human').textContent = 0;
    document.getElementById('score-ai').textContent = 0;
    document.getElementById('score-draw').textContent = 0;
    moveCount = 1;
    document.getElementById('history-list').innerHTML = `
        <div class="history-header">
            <span>#</span>
            <span>Minimax</span>
            <span>Alpha-Beta</span>
            <span>Saved</span>
        </div>
    `;
    document.getElementById('bar-section').style.display = 'none';
    ['mm-time', 'mm-nodes', 'ab-time', 'ab-nodes', 'comp-speedup', 'comp-nodes-saved', 'comp-prune-pct'].forEach(id => {
        document.getElementById(id).textContent = "—";
    });
}

// ─── Grid Background ──────────────────────────────────────────
const bgCanvas = document.getElementById("bg-canvas");
const bgCtx = bgCanvas.getContext("2d");

let bgWidth = window.innerWidth;
let bgHeight = window.innerHeight;

function resizeAndDraw() {
    bgWidth = window.innerWidth;
    bgHeight = window.innerHeight;
    bgCanvas.width = bgWidth;
    bgCanvas.height = bgHeight;
    drawGrid();
}

window.addEventListener("resize", resizeAndDraw);

function drawGrid() {
    bgCtx.clearRect(0, 0, bgWidth, bgHeight);
    
    // Draw grid dots exactly like navigation.js
    bgCtx.fillStyle = "rgba(129,140,248,0.04)";
    for (let gx = 30; gx < bgWidth; gx += 40) {
        for (let gy = 30; gy < bgHeight; gy += 40) {
            bgCtx.beginPath();
            bgCtx.arc(gx, gy, 1, 0, Math.PI * 2);
            bgCtx.fill();
        }
    }
}

// Initial draw
resizeAndDraw();
