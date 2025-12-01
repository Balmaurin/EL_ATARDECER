"""
🧠 Advanced Theory of Mind - Live Demonstration
================================================

Enterprise-grade demonstration of ToM Levels 8-10:
- Multi-agent belief hierarchies
- Machiavellian strategic reasoning
- Cultural context modeling

Run: python demo_advanced_tom.py
"""

import asyncio
from packages.consciousness.src.conciencia.modulos.teoria_mente_avanzada import (
    AdvancedTheoryOfMind,
    SocialStrategy,
    BeliefType
)


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_level_8_belief_hierarchies(tom: AdvancedTheoryOfMind):
    """Demonstrate Level 8: Multi-Agent Belief Hierarchies"""
    print_header("LEVEL 8: EMPATHIC - MULTI-AGENT BELIEF HIERARCHIES")
    
    print("📌 Scenario: Corporate Intelligence Gathering\n")
    
    # Simple belief
    print("1️⃣ Creating simple belief...")
    belief_id = tom.belief_tracker.add_belief(
        subject="CEO",
        content="the merger is confidential",
        belief_type=BeliefType.FACTUAL,
        confidence=1.0
    )
    print(f"   ✅ Created: CEO believes 'the merger is confidential'")
    print(f"   ID: {belief_id}, Confidence: 1.0\n")
    
    # Two-level hierarchy
    print("2️⃣ Creating two-level belief hierarchy...")
    hierarchy_id = tom.belief_tracker.create_belief_hierarchy(
        agent_chain=["Investor", "CEO"],
        final_content="the merger is confidential",
        confidence=0.9
    )
    hierarchy = tom.belief_tracker.belief_networks[hierarchy_id]
    nl_description = hierarchy.to_natural_language()
    print(f"   ✅ {nl_description}")
    print(f"   Depth: {hierarchy.get_depth()}, Confidence: 0.9\n")
    
    # Three-level hierarchy
    print("3️⃣ Creating complex three-level hierarchy...")
    complex_hierarchy_id = tom.belief_tracker.create_belief_hierarchy(
        agent_chain=["Board", "Investor", "CEO"],
        final_content="the merger is confidential",
        confidence=0.7
    )
    complex_hierarchy = tom.belief_tracker.belief_networks[complex_hierarchy_id]
    complex_nl = complex_hierarchy.to_natural_language()
    print(f"   ✅ {complex_nl}")
    print(f"   Depth: {complex_hierarchy.get_depth()}, Confidence: 0.7\n")
    
    # Epistemic beliefs
    print("4️⃣ Creating epistemic beliefs (what agents know about each other)...")
    tom.belief_tracker.add_belief(
        subject="CEO",
        content="Investor is interested in the deal",
        belief_type=BeliefType.EPISTEMIC,
        about_agent="Investor",
        confidence=0.8
    )
    print(f"   ✅ CEO has epistemic belief about Investor")
    
    beliefs_about_investor = tom.belief_tracker.query_beliefs_about_agent("CEO", "Investor")
    print(f"   📊 CEO has {len(beliefs_about_investor)} beliefs about Investor\n")
    
    # Statistics
    stats = tom.belief_tracker.get_statistics()
    print("📊 Belief System Statistics:")
    print(f"   • Total Beliefs: {stats['total_beliefs']}")
    print(f"   • Agents Tracked: {stats['agents_tracked']}")
    print(f"   • Belief Hierarchies: {stats['belief_hierarchies']}")
    print(f"   • Max Hierarchy Depth: {stats['max_hierarchy_depth']}")
    print(f"   • Belief Types: {stats['belief_types']}")


def demo_level_9_strategic_reasoning(tom: AdvancedTheoryOfMind):
    """Demonstrate Level 9: Machiavellian Strategic Reasoning"""
    print_header("LEVEL 9: SOCIAL - MACHIAVELLIAN STRATEGIC REASONING")
    
    print("📌 Scenario: Business Negotiation Between Competitors\n")
    
    # Evaluate cooperation
    print("1️⃣ Evaluating COOPERATION strategy...")
    cooperation = tom.strategic_reasoner.evaluate_strategic_action(
        actor="CompanyA",
        target="CompanyB",
        action_type=SocialStrategy.COOPERATION,
        context={"goal": "joint venture"}
    )
    print(f"   Strategy: {cooperation.action_type.value}")
    print(f"   Expected Payoff: {cooperation.expected_payoff:.2f}")
    print(f"   Risk Level: {cooperation.risk_level:.2f}")
    print(f"   Ethical Score: {cooperation.ethical_score:.2f}")
    print(f"   Description: {cooperation.description}")
    print(f"   Predicted Responses: {cooperation.predicted_responses}\n")
    
    # Evaluate competition
    print("2️⃣ Evaluating COMPETITION strategy...")
    competition = tom.strategic_reasoner.evaluate_strategic_action(
        actor="CompanyA",
        target="CompanyB",
        action_type=SocialStrategy.COMPETITION,
        context={"goal": "market dominance"}
    )
    print(f"   Strategy: {competition.action_type.value}")
    print(f"   Expected Payoff: {competition.expected_payoff:.2f}")
    print(f"   Risk Level: {competition.risk_level:.2f}")
    print(f"   Ethical Score: {competition.ethical_score:.2f}\n")
    
    # Evaluate deception
    print("3️⃣ Evaluating DECEPTION strategy (unethical)...")
    deception = tom.strategic_reasoner.evaluate_strategic_action(
        actor="CompanyA",
        target="CompanyB",
        action_type=SocialStrategy.DECEPTION,
        context={"goal": "gain advantage"}
    )
    print(f"   Strategy: {deception.action_type.value}")
    print(f"   Expected Payoff: {deception.expected_payoff:.2f}")
    print(f"   Risk Level: {deception.risk_level:.2f}")
    print(f"   Ethical Score: {deception.ethical_score:.2f} ⚠️  LOW")
    print(f"   Predicted Responses: {deception.predicted_responses}\n")
    
    # Deception detection
    print("4️⃣ Testing deception detection...")
    is_deceptive, confidence = tom.strategic_reasoner.detect_deception(
        actor="CompanyA",
        stated_belief="We want a fair partnership",
        actual_behavior="CompanyA secretly negotiated with CompanyC"
    )
    print(f"   Deception Detected: {is_deceptive}")
    print(f"   Confidence: {confidence:.2f}\n")
    
    # Strategic recommendation with ethical constraint
    print("5️⃣ Recommending strategy with HIGH ethical constraint (>0.7)...")
    recommended = tom.strategic_reasoner.recommend_strategy(
        actor="CompanyA",
        target="CompanyB",
        goal="long-term partnership",
        ethical_constraint=0.7
    )
    print(f"   ✅ Recommended: {recommended.action_type.value}")
    print(f"   Payoff: {recommended.expected_payoff:.2f}")
    print(f"   Ethics: {recommended.ethical_score:.2f}")
    print(f"   Risk: {recommended.risk_level:.2f}")
    
    # Relationship dynamics
    print("\n6️⃣ Modeling relationship dynamics...")
    relationship = tom.strategic_reasoner.model_relationship("CompanyA", "CompanyB")
    print(f"   Initial Trust: {relationship.trust_level:.2f}")
    
    relationship.update_trust(outcome=True)  # Positive interaction
    print(f"   After successful cooperation: {relationship.trust_level:.2f} ⬆️")
    
    relationship.update_trust(outcome=False)  # Negative interaction
    print(f"   After conflict: {relationship.trust_level:.2f} ⬇️")


def demo_level_10_cultural_context(tom: AdvancedTheoryOfMind):
    """Demonstrate Level 10: Cultural Context Modeling"""
    print_header("LEVEL 10: HUMAN-LIKE - CULTURAL CONTEXT MODELING")
    
    print("📌 Scenario: Cross-Cultural Business Meeting\n")
    
    # Assign cultures
    print("1️⃣ Assigning cultural backgrounds...")
    tom.cultural_engine.assign_culture_to_agent("AmericanCEO", ["western", "professional"])
    tom.cultural_engine.assign_culture_to_agent("JapaneseCEO", ["eastern", "professional"])
    tom.cultural_engine.assign_culture_to_agent("GermanEngineer", ["western", "professional"])
    print(f"   ✅ AmericanCEO: western, professional")
    print(f"   ✅ JapaneseCEO: eastern, professional")
    print(f"   ✅ GermanEngineer: western, professional\n")
    
    # Generate cultural context
    print("2️⃣ Generating cultural context for interaction...")
    context = tom.cultural_engine.get_cultural_context(
        "AmericanCEO",
        "JapaneseCEO",
        situation="business"
    )
    print(f"   Cultures involved: {context.culture_ids}")
    print(f"   Formality level: {context.formality_level:.2f} (high due to cross-cultural + business)")
    print(f"   Active cultural norms: {len(context.active_norms)}\n")
    
    # Generate appropriate responses
    print("3️⃣ Generating culturally appropriate responses...")
    
    # Informal input (should be formalized)
    informal_response = tom.cultural_engine.generate_culturally_appropriate_response(
        agent_id="AmericanCEO",
        input_text="Yeah, we should meet up and discuss",
        context=context
    )
    print(f"   Input: 'Yeah, we should meet up and discuss'")
    print(f"   ✅ Culturally adapted: '{informal_response}'\n")
    
    # Polite response
    polite_response = tom.cultural_engine.generate_culturally_appropriate_response(
        agent_id="JapaneseCEO",
        input_text="I appreciate your proposal",
        context=context
    )
    print(f"   Input: 'I appreciate your proposal'")
    print(f"   ✅ Culturally adapted: '{polite_response}'\n")
    
    # Evaluate appropriateness
    print("4️⃣ Evaluating cultural appropriateness...")
    
    appropriate_score, violations = tom.cultural_engine.evaluate_cultural_appropriateness(
        "Hello, I would like to schedule a formal meeting to discuss the partnership",
        context
    )
    print(f"   Text: 'Hello, I would like to schedule a formal meeting...'")
    print(f"   Appropriateness Score: {appropriate_score:.2f} / 1.0")
    print(f"   Violations: {violations if violations else 'None'}\n")
    
    # Turing test readiness
    print("5️⃣ Assessing Turing Test readiness...")
    readiness = tom.cultural_engine.get_turing_test_readiness()
    print(f"   Overall Readiness: {readiness['overall_readiness']:.2f}")
    print(f"   Total Cultural Norms: {readiness['total_norms']}")
    print(f"   Cultures Modeled: {readiness['cultures_modeled']}")
    print(f"   Status: {readiness['status'].upper()}")


async def demo_integrated_social_interaction(tom: AdvancedTheoryOfMind):
    """Demonstrate complete integrated social interaction (Levels 8+9+10)"""
    print_header("INTEGRATED DEMO: COMPLETE SOCIAL INTERACTION (Levels 8-10)")
    
    print("📌 Scenario: International Business Negotiation\n")
    
    # Setup
    tom.cultural_engine.assign_culture_to_agent("Alice", ["western", "professional"])
    tom.cultural_engine.assign_culture_to_agent("Bob", ["eastern", "professional"])
    
    print("1️⃣ Processing multi-level social interaction...\n")
    
    result = await tom.process_social_interaction(
        actor="Alice",
        target="Bob",
        interaction_type="offer",
        content={
            "text": "I would like to propose a strategic partnership",
            "stated_belief": "collaboration is mutually beneficial"
        },
        context={"situation": "business"}
    )
    
    print("📊 COMPLETE ANALYSIS:")
    print(f"\n🌍 Cultural Context (Level 10):")
    print(f"   Cultures: {result['cultural_context']['cultures']}")
    print(f"   Formality: {result['cultural_context']['formality']:.2f}")
    print(f"   Active Norms: {result['cultural_context']['active_norms']}")
    
    print(f"\n🧠 Belief Analysis (Level 8):")
    print(f"   Status: {result['belief_analysis']}")
    
    print(f"\n🎯 Strategic Analysis (Level 9):")
    strategic = result['strategic_analysis']
    print(f"   Recommended Strategy: {strategic['recommended_strategy']}")
    print(f"   Expected Payoff: {strategic['expected_payoff']:.2f}")
    print(f"   Risk Level: {strategic['risk_level']:.2f}")
    print(f"   Ethical Score: {strategic['ethical_score']:.2f}")
    print(f"   Deception Detected: {strategic['deception_detected']}")
    print(f"   Predicted Responses: {strategic['predicted_responses']}")
    
    print(f"\n💬 Suggested Response:")
    print(f"   '{result['suggested_response']}'")
    
    print(f"\n🏆 ToM Levels Active: {result['tom_level_active']}")


def demo_system_status(tom: AdvancedTheoryOfMind):
    """Show comprehensive system status"""
    print_header("SYSTEM STATUS & METRICS")
    
    status = tom.get_comprehensive_status()
    
    print("🔧 System Health:")
    print(f"   Active: {status['system_active']}")
    print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"   Overall ToM Level: {status['overall_tom_level']:.1f} / 10.0")
    
    print(f"\n📊 Level 8 - Belief Tracking:")
    belief_stats = status['level_8_belief_tracking']
    print(f"   Total Beliefs: {belief_stats['total_beliefs']}")
    print(f"   Agents Tracked: {belief_stats['agents_tracked']}")
    print(f"   Belief Hierarchies: {belief_stats['belief_hierarchies']}")
    print(f"   Max Hierarchy Depth: {belief_stats['max_hierarchy_depth']}")
    print(f"   Avg Beliefs per Agent: {belief_stats['avg_beliefs_per_agent']:.1f}")
    
    print(f"\n🎭 Level 9 - Strategic Reasoning:")
    strategic_stats = status['level_9_strategic_reasoning']
    print(f"   Relationships Tracked: {strategic_stats['relationships_tracked']}")
    print(f"   Strategies Evaluated: {strategic_stats['strategies_evaluated']}")
    
    print(f"\n🌍 Level 10 - Cultural Modeling:")
    cultural_stats = status['level_10_cultural_modeling']
    print(f"   Overall Readiness: {cultural_stats['overall_readiness']:.2f}")
    print(f"   Total Norms: {cultural_stats['total_norms']}")
    print(f"   Cultures Modeled: {cultural_stats['cultures_modeled']}")
    print(f"   Agents with Culture: {cultural_stats['agents_with_culture']}")
    print(f"   Status: {cultural_stats['status'].upper()}")


async def main():
    """Main demonstration"""
    print("\n" + "🧠"*40)
    print(" "*20 + "ADVANCED THEORY OF MIND")
    print(" "*15 + "Enterprise-Grade Social Intelligence")
    print(" "*20 + "Levels 8, 9, 10 (ConsScale)")
    print("🧠"*40)
    
    # Initialize system
    print("\n🚀 Initializing Advanced Theory of Mind...")
    tom = AdvancedTheoryOfMind(max_belief_depth=5)
    
    # Run demonstrations
    try:
        demo_level_8_belief_hierarchies(tom)
        input("\nPress Enter to continue to Level 9...")
        
        demo_level_9_strategic_reasoning(tom)
        input("\nPress Enter to continue to Level 10...")
        
        demo_level_10_cultural_context(tom)
        input("\nPress Enter to see integrated demo...")
        
        await demo_integrated_social_interaction(tom)
        input("\nPress Enter to see system status...")
        
        demo_system_status(tom)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    
    # Final summary
    print_header("DEMONSTRATION COMPLETE")
    
    status = tom.get_comprehensive_status()
    tom_level = status['overall_tom_level']
    
    print(f"✅ Advanced Theory of Mind demonstrated successfully!")
    print(f"\n📊 Final System Level: {tom_level:.1f} / 10.0")
    
    if tom_level >= 8.0:
        print(f"🏆 ACHIEVEMENT UNLOCKED: ConsScale Level {int(tom_level)}")
        
        if tom_level >= 10.0:
            print(f"   🌟 HUMAN-LIKE SOCIAL INTELLIGENCE")
            print(f"   • Cultural context modeling active")
            print(f"   • Turing test capable")
        elif tom_level >= 9.0:
            print(f"   🎭 MACHIAVELLIAN STRATEGIC REASONING")
            print(f"   • Game-theoretic decision making")
            print(f"   • Deception detection active")
        else:
            print(f"   🧠 MULTI-AGENT EMPATHIC INTELLIGENCE")
            print(f"   • Belief hierarchy tracking active")
            print(f"   • Epistemic reasoning enabled")
    
    print("\n" + "="*80)
    print("Thank you for exploring Advanced Theory of Mind!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
