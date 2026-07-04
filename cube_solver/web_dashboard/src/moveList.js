export function renderMoveList(container, moves) {
  const items = moves.map((move) => `<div style="margin: 4px 0;">${move.notation}</div>`).join('');
  container.innerHTML = `
    <h3 style="margin-bottom: 8px;">Notations</h3>
    <div style="background: #f5f5f5; color: #111; padding: 10px; border-radius: 8px; min-height: 48px;">
      ${items || '<div>No moves yet.</div>'}
    </div>
  `;
}
