import type { TraceTurn } from '../../types/trace';
import TraceStepList from './TraceStepList';

interface Props {
  turns: TraceTurn[];
  defaultOpenLast?: boolean;
  expandBodies?: boolean;
}

export default function TraceTurnList({
  turns,
  defaultOpenLast = true,
  expandBodies = false,
}: Props) {
  if (turns.length === 0) {
    return null;
  }

  return (
    <div className="cursor-trace-turn-list">
      {turns.map((turn, turnIndex) => {
        if (turn.steps.length === 0) {
          return null;
        }
        const openLast =
          defaultOpenLast &&
          turnIndex === turns.length - 1 &&
          (turn.live === true || turnIndex === turns.length - 1);

        return (
          <section
            key={turn.id}
            className={`cursor-trace-turn${turn.live ? ' cursor-trace-turn--live' : ''}`}
          >
            <header className="cursor-trace-turn-header">
              <span className="cursor-trace-turn-label">{turn.label}</span>
              <span className="cursor-trace-turn-meta">{turn.steps.length} 步</span>
            </header>
            <TraceStepList
              steps={turn.steps}
              defaultOpenLast={openLast}
              expandBodies={expandBodies}
            />
          </section>
        );
      })}
    </div>
  );
}
