"""
Demonstration of CIAF LCM Protocol Interface Usage

This script demonstrates how to use the refactored LCM system with
protocol interfaces for dependency injection and better architecture.
"""

from ciaf.lcm import (
    LCMDatasetManager, DatasetMetadata, DatasetSplit,
    DefaultRNG, DefaultMerkle, DefaultAnchorDeriver, 
    InMemoryAnchorStore, DefaultSigner, LCMPolicy
)


def demo_protocol_dependency_injection():
    """Demonstrate protocol-based dependency injection."""
    print("🔬 CIAF LCM Protocol Interface Demonstration")
    print("=" * 60)
    
    # 1. Create custom protocol implementations
    print("\n1️⃣ Creating Protocol Implementations...")
    custom_rng = DefaultRNG()
    custom_anchor_deriver = DefaultAnchorDeriver()
    custom_anchor_store = InMemoryAnchorStore()
    custom_signer = DefaultSigner("demo_key")
    
    # Custom Merkle factory
    def custom_merkle_factory(leaves=None):
        print(f"   🌳 Creating Merkle tree with {len(leaves or [])} leaves")
        return DefaultMerkle(leaves)
    
    print(f"   ✅ RNG: {type(custom_rng).__name__}")
    print(f"   ✅ Anchor Deriver: {type(custom_anchor_deriver).__name__}")
    print(f"   ✅ Anchor Store: {type(custom_anchor_store).__name__}")
    print(f"   ✅ Signer: {type(custom_signer).__name__} (key: {custom_signer.key_id})")
    print(f"   ✅ Merkle Factory: Custom implementation")
    
    # 2. Create LCM policy with injected protocols
    print("\n2️⃣ Creating LCM Policy with Injected Protocols...")
    policy = LCMPolicy(
        rng=custom_rng,
        anchor_deriver=custom_anchor_deriver,
        anchor_store=custom_anchor_store,
        signer=custom_signer,
        merkle_factory=custom_merkle_factory
    )
    print(f"   ✅ Policy created with injected protocols")
    
    # 3. Create dataset manager with policy
    print("\n3️⃣ Creating Dataset Manager with Protocol Policy...")
    dataset_manager = LCMDatasetManager(policy=policy)
    
    # 4. Create dataset metadata
    print("\n4️⃣ Creating Dataset with Protocol-Based Anchors...")
    metadata = DatasetMetadata(
        name="protocol_demo_dataset",
        description="Demonstrates protocol interface usage",
        features=["feature_1", "feature_2", "feature_3"],
        total_samples=100
    )
    
    # Create dataset splits using protocol implementations
    splits = dataset_manager.create_dataset_splits(
        dataset_id="protocol_demo",
        metadata=metadata,
        master_password="demo_password",
        splits=[DatasetSplit.TRAIN, DatasetSplit.TEST]
    )
    
    print(f"   ✅ Created {len(splits)} dataset splits using protocols")
    
    # 5. Demonstrate protocol usage
    print("\n5️⃣ Demonstrating Protocol Features...")
    
    train_anchor = splits[DatasetSplit.TRAIN]
    
    # Show RNG usage
    random_data = train_anchor.rng.random_bytes(16)
    print(f"   🎲 RNG generated {len(random_data)} random bytes")
    
    # Show commitment creation using protocols
    commitment = train_anchor.create_commitment("test_data")
    print(f"   🔐 Created commitment: {commitment}")
    
    # Add sample and show Merkle tree usage
    sample_hash_1 = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"  # Valid hex
    sample_hash_2 = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"  # Valid hex
    train_anchor.add_sample_hash(sample_hash_1)
    train_anchor.add_sample_hash(sample_hash_2)
    merkle_root = train_anchor.get_merkle_root()
    print(f"   🌳 Merkle root after adding samples: {merkle_root[:16]}...")
    
    # Show anchor store usage
    anchor_data = train_anchor.to_dict()
    policy.anchor_store.append_anchor(anchor_data)
    latest = policy.anchor_store.get_latest_anchor()
    print(f"   📚 Stored anchor in store, latest anchor ID: {latest.get('dataset_id', 'N/A')}")
    
    # 6. Show protocol swappability
    print("\n6️⃣ Demonstrating Protocol Swappability...")
    
    # Create new policy with different implementations
    new_policy = LCMPolicy()  # Uses defaults
    print(f"   🔄 Created new policy with default protocols")
    
    # Show they're different instances
    print(f"   📊 Original RNG: {id(policy.rng)}")
    print(f"   📊 New RNG: {id(new_policy.rng)}")
    print(f"   ✅ Different protocol instances can be swapped easily")
    
    print(f"\n🎉 Protocol Interface Demonstration Complete!")
    print(f"   - Dependency injection working correctly")
    print(f"   - Protocol interfaces provide type safety")
    print(f"   - Easy to swap implementations for testing/customization")
    print(f"   - Clean separation of concerns")


def demo_protocol_type_safety():
    """Demonstrate protocol type safety."""
    print("\n" + "=" * 60)
    print("🛡️ Protocol Type Safety Demonstration")
    print("=" * 60)
    
    from ciaf.core.interfaces import RNG, Merkle, AnchorDeriver, AnchorStore, Signer
    
    # Show that our implementations satisfy the protocols
    rng = DefaultRNG()
    merkle = DefaultMerkle()
    deriver = DefaultAnchorDeriver()
    store = InMemoryAnchorStore()
    signer = DefaultSigner()
    
    print(f"✅ DefaultRNG implements RNG protocol: {isinstance(rng, RNG)}")
    print(f"✅ DefaultMerkle implements Merkle protocol: {isinstance(merkle, Merkle)}")
    print(f"✅ DefaultAnchorDeriver implements AnchorDeriver protocol: {isinstance(deriver, AnchorDeriver)}")
    print(f"✅ InMemoryAnchorStore implements AnchorStore protocol: {isinstance(store, AnchorStore)}")
    print(f"✅ DefaultSigner implements Signer protocol: {isinstance(signer, Signer)}")
    
    print(f"\n🎯 All implementations correctly satisfy their protocol contracts!")


if __name__ == "__main__":
    demo_protocol_dependency_injection()
    demo_protocol_type_safety()