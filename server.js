const express = require('express');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

const DB_PATH = path.join(__dirname, 'spatial-db.json');

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

function readSpatialDB() {
    if (!fs.existsSync(DB_PATH)) {
        return { spaceId: "lab-space", anchors: {} };
    }
    try {
        return JSON.parse(fs.readFileSync(DB_PATH, 'utf8'));
    } catch {
        return { spaceId: "lab-space", anchors: {} };
    }
}

app.get('/api/spatial/map', (req, res) => res.json(readSpatialDB()));

io.on('connection', (socket) => {
    socket.emit('spatialAnchorsUpdated', readSpatialDB());
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`Node server running on port ${PORT}`));