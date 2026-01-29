# UFO Galaxy Dashboard v5.0

A futuristic, cyberpunk-inspired dashboard for monitoring the UFO Galaxy distributed AI computing network.

## Features

- **Real-time Node Status Monitoring**: Track all 102 nodes with live CPU, memory, and network usage
- **Interactive Network Topology**: Visualize node connections with zoom, pan, and animated data flow
- **Performance Analytics**: Charts for CPU, memory, network traffic, and task throughput
- **Task Queue Management**: Monitor pending, running, completed, and failed tasks
- **System Log Viewer**: Real-time log stream with filtering and search capabilities
- **Dark Theme with Neon Accents**: Deep space color scheme with glowing UI elements

## Tech Stack

- React 18 + TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- Recharts for data visualization
- WebSocket for real-time updates
- React Router for navigation

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Development

The dashboard will be available at `http://localhost:3000`

## Project Structure

```
dashboard/
├── public/
│   └── ufo-icon.svg
├── src/
│   ├── components/
│   │   ├── NodeStatus.tsx       # Node grid and status monitoring
│   │   ├── NetworkTopology.tsx  # Interactive network visualization
│   │   ├── PerformanceCharts.tsx # Analytics charts
│   │   ├── TaskQueue.tsx        # Task management interface
│   │   └── LogViewer.tsx        # Log stream viewer
│   ├── hooks/
│   │   └── useWebSocket.ts      # WebSocket connection hook
│   ├── types/
│   │   └── index.ts             # TypeScript type definitions
│   ├── App.tsx                  # Main application component
│   ├── index.css                # Global styles
│   └── main.tsx                 # Entry point
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── README.md
```

## WebSocket Integration

The dashboard uses WebSocket for real-time updates. In development mode, it uses simulated data. For production, connect to your WebSocket server:

```typescript
const { isConnected, sendMessage } = useWebSocket('ws://your-server:8080/ws', {
  onNodeUpdate: (node) => console.log('Node updated:', node),
  onTaskUpdate: (task) => console.log('Task updated:', task),
  // ... other handlers
});
```

## Customization

### Theme Colors

Edit `src/index.css` to customize the color scheme:

```css
:root {
  --bg-primary: #0A0E17;      /* Deep space background */
  --neon-blue: #3B82F6;       /* Primary accent */
  --neon-purple: #8B5CF6;     /* Secondary accent */
  /* ... */
}
```

### Adding New Components

1. Create component in `src/components/`
2. Add route in `src/App.tsx`
3. Add navigation item in the Sidebar

## License

MIT License - UFO Galaxy Project
