export class DashboardSocket {
  constructor({ onOpen, onMessage, onClose, url }) {
    this.url = url;
    this.onOpen = onOpen;
    this.onMessage = onMessage;
    this.onClose = onClose;
    this.socket = null;
    this.retryTimer = null;
  }

  connect() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener('open', () => this.onOpen?.());
    this.socket.addEventListener('message', (event) => this.onMessage?.(JSON.parse(event.data)));
    this.socket.addEventListener('close', () => {
      this.onClose?.();
      this.retryTimer = setTimeout(() => this.connect(), 1000);
    });
  }

  disconnect() {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
    }
    if (this.socket) {
      this.socket.close();
    }
  }
}
