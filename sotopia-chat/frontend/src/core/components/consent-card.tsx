import { Button } from "@/components/ui/button";

interface ConsentCardProps {
    onAccept: () => void;
}

export function ConsentCard({ onAccept }: ConsentCardProps) {
    return (
        <div className="mx-auto max-w-3xl rounded-lg border bg-card p-8 shadow-sm">
            <h1 className="text-3xl font-light">
                Sotopia Research Participation Consent
            </h1>
            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
                You are invited to participate in a research study aiming to
                study realistic social interactions between humans and AI
                agents. You will take the role of a character within a social
                scenario and interact with other agents to achieve assigned
                goals. Participation is voluntary; you may withdraw at any time.
                Please ensure you are at least 18 years old.
            </p>
            <p className="mt-4 rounded bg-muted p-4 text-sm leading-relaxed">
                <strong>Confidentiality:</strong> Your responses may be used for
                research analysis. Data will be stored securely and shared only
                with qualified researchers. If you wish to withdraw your data
                later, contact the research team with your session identifier.
            </p>
            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
                For questions about your rights as a participant, contact the CMU
                Office of Research Integrity and Compliance (STUDY2023_00000299,
                irb-review@andrew.cmu.edu, 412-268-4721).
            </p>
            <div className="mt-6 flex items-center justify-between">
                <Button variant="link" onClick={() => window.print()}>
                    Print a copy of this consent form
                </Button>
                <Button onClick={onAccept}>I agree and wish to continue</Button>
            </div>
        </div>
    );
}
