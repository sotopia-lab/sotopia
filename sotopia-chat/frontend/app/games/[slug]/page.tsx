import { notFound } from "next/navigation";
import { GameExperience } from "@/core/components/game-experience";
import { getGameBySlug } from "@/core/config/games";

interface GamePageProps {
    params: { slug: string };
}

export default function GamePage({ params }: GamePageProps) {
    const game = getGameBySlug(params.slug);

    if (!game) {
        notFound();
    }

    return <GameExperience slug={params.slug} />;
}
